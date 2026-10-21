"""Independent small calculations and actual model paths for lessons 13–16."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from backbone import ConvNeXt, PVT, ViT


class ModernBackboneLessonTests(unittest.TestCase):
    def test_attention_example_matches_actual_module_and_stable_softmax(self):
        x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
        layer = ViT.Attention(2, heads=1, dim_head=2).double().eval()
        with torch.no_grad():
            layer.to_qkv.weight.copy_(torch.eye(2).repeat(3, 1))
        scores = x @ x.transpose(-1, -2) / 2 ** .5
        weights = scores.softmax(-1)
        expected = torch.tensor([.4011120926797859, .1977758146404282,
                                 .4011120926797859], dtype=torch.float64)
        self.assertTrue(torch.allclose(weights[0, 0], expected, atol=1e-12))
        self.assertTrue(torch.allclose(layer(x), weights @ x, atol=1e-12))
        self.assertAlmostEqual(layer(x)[0, 0, 0].item(), .8022241853595719)
        self.assertAlmostEqual(layer(x)[0, 0, 1].item(), .5988879073202141)
        shifted_exp = (scores - scores.amax(-1, keepdim=True)).exp()
        self.assertTrue(torch.allclose(shifted_exp / shifted_exp.sum(-1, keepdim=True), weights))
        self.assertFalse(torch.allclose(weights, weights.transpose(-1, -2)))
        self.assertTrue(torch.allclose(scores, scores.transpose(-1, -2)))

    def test_attention_dropout_does_not_renormalize_each_row(self):
        layer = ViT.Attention(2, heads=1, dim_head=2, dropout=.5).double()
        with torch.no_grad():
            layer.to_qkv.weight.copy_(torch.eye(2).repeat(3, 1))
        traces = []
        hook = layer.dropout.register_forward_hook(
            lambda _m, inputs, output: traces.append((inputs[0].detach(), output.detach())))
        try:
            with torch.random.fork_rng(), torch.no_grad():
                torch.manual_seed(0)
                x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
                layer.train()(x)
                layer.eval()(x)
        finally:
            hook.remove()
        before, after = traces[0]
        self.assertTrue(torch.allclose(before.sum(-1), torch.ones_like(before.sum(-1))))
        self.assertTrue(torch.all((after == 0) | (after == 2 * before)))
        self.assertFalse(torch.allclose(after.sum(-1), torch.ones_like(after.sum(-1))))
        self.assertTrue(torch.equal(*traces[1]))

    def test_patch_flattening_interleaves_rgb_and_keeps_row_major_patch_order(self):
        layer = ViT.ViT(image_size=2, patch_size=2, num_classes=10, dim=6,
                        depth=1, heads=1, dim_head=6, mlp_dim=12).to_patch_embedding[0]
        rgb = torch.arange(1, 13.).reshape(1, 2, 2, 3).permute(0, 3, 1, 2)
        self.assertTrue(torch.equal(layer(rgb).flatten(), torch.arange(1, 13.)))
        self.assertEqual(rgb.flatten().tolist(), [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12])
        grid = torch.arange(16.).reshape(1, 1, 4, 4).expand(1, 3, 4, 4)
        self.assertEqual(layer(grid)[0, :, 0].tolist(), [0, 2, 8, 10])

    def test_vit_factories_and_input_parameters_use_the_documented_shapes(self):
        variants = [('tiny', 192, 12, 3, 768), ('small', 384, 12, 6, 1536),
                    ('base', 768, 12, 12, 3072), ('large', 1024, 24, 16, 4096)]
        for name, dim, depth, heads, hidden in variants:
            with self.subTest(name=name), torch.device('meta'):
                model = getattr(ViT, f'vit_{name}')().eval()
                self.assertEqual(len(model.transformer.layers), depth)
                attention = model.transformer.layers[0][0].fn
                self.assertEqual(attention.heads, heads)
                self.assertEqual(attention.to_qkv.out_features, heads * 64 * 3)
                self.assertEqual(model.transformer.layers[0][1].fn.net[0].out_features, hidden)
                self.assertEqual(tuple(model(torch.empty(2, 3, 32, 32)).shape), (2, 10))
                self.assertEqual(tuple(model.pos_embedding.shape), (1, 65, dim))
                if name == 'tiny':
                    self.assertEqual(sum(p.numel() for p in model.to_patch_embedding.parameters()), 9408)
                    self.assertEqual(model.pos_embedding.numel(), 12480)
                    self.assertEqual(model.cls_token.numel(), 192)
                    self.assertEqual(tuple(attention.attend(torch.empty(2, 3, 65, 65)).shape), (2, 3, 65, 65))

    def test_single_kv_attention_is_shared_but_residual_retains_differences(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            block = PVT.Block(dim=4, num_heads=1, sr_ratio=2).double().eval()
        # Zero only the test fixture's MLP output to isolate the existing residual path.
        with torch.no_grad():
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
        x = torch.tensor([[[1., 0., 2., -1.], [0., 1., -2., 3.],
                           [2., 1., 0., 1.], [-2., 0., 1., 2.]]], dtype=torch.float64)
        branch = block.attn(block.norm1(x), 2, 2)
        self.assertTrue(torch.allclose(branch, branch[:, :1].expand_as(branch), atol=1e-12))
        self.assertTrue(torch.allclose(block(x, 2, 2), x + branch, atol=1e-12))
        self.assertFalse(torch.allclose(block(x, 2, 2)[:, 0], block(x, 2, 2)[:, 1]))
        branch.sum().backward()
        self.assertEqual(torch.count_nonzero(block.attn.q.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(block.attn.kv.weight.grad[:4]).item(), 0)
        self.assertGreater(torch.count_nonzero(block.attn.kv.weight.grad[4:]).item(), 0)
        self.assertTrue(torch.equal(torch.tensor([[3.], [-6.]]).softmax(-1), torch.ones(2, 1)))

    def test_pvt_sra_actual_shapes_at_small_and_paper_resolutions(self):
        linspace = torch.linspace
        # Only the scalar DropPath schedule stays on CPU; all model tensors are meta.
        with patch('torch.linspace', side_effect=lambda *a, **k: linspace(*a, **{**k, 'device': 'cpu'})):
            for size, keys in ((32, [16, 16, 4, 1]), (224, [49, 49, 49, 49])):
                with self.subTest(size=size), torch.device('meta'):
                    model = PVT.pvt_tiny(img_size=size).eval()
                    traces = []
                    modules = [model.block1[0].attn, model.block2[0].attn,
                               model.block3[0].attn, model.block4[0].attn]
                    hooks = [m.attn_drop.register_forward_hook(
                        lambda _m, _i, o: traces.append(tuple(o.shape))) for m in modules]
                    try:
                        self.assertEqual(tuple(model(torch.empty(2, 3, size, size)).shape), (2, 10))
                    finally:
                        for hook in hooks:
                            hook.remove()
                    self.assertEqual(traces, [(2, h, (size // (4 * 2 ** i)) ** 2, k)
                                              for i, (h, k) in enumerate(zip([1, 2, 5, 8], keys))])

    def test_convnext_layernorm_axes_affine_and_constant_vector(self):
        x = torch.tensor([[[[1., 3., 5.]], [[3., 7., 5.]]]], dtype=torch.float64)
        first = ConvNeXt.LayerNorm(2, data_format='channels_first').double()
        last = ConvNeXt.LayerNorm(2).double()
        normalized = first(x)
        self.assertTrue(torch.allclose(normalized, last(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        self.assertAlmostEqual(normalized[0, 0, 0, 0].item(), -.999999500000375)
        self.assertTrue(torch.equal(normalized[0, :, 0, 2], torch.zeros(2, dtype=torch.float64)))
        with torch.no_grad():
            first.weight.copy_(torch.tensor([2., .5]))
            first.bias.copy_(torch.tensor([1., -1.]))
        expected = normalized * torch.tensor([2., .5])[:, None, None] + torch.tensor([1., -1.])[:, None, None]
        self.assertTrue(torch.allclose(first(x), expected))

    def test_convnext_block_shape_weights_and_padding_coverage(self):
        with torch.device('meta'):
            block = ConvNeXt.Block(96)
            self.assertEqual(block.dwconv.weight.numel(), 4704)
            self.assertEqual(block.pwconv1.weight.numel() + block.pwconv2.weight.numel(), 73728)
            self.assertEqual(sum(p.numel() for p in block.parameters()), 79296)
            self.assertEqual(tuple(block(torch.empty(2, 96, 8, 8)).shape), (2, 96, 8, 8))
            self.assertIsInstance(block.drop_path, nn.Identity)
        for side, row, col, expected in ((8, 4, 4, 49), (8, 0, 0, 16), (1, 0, 0, 1)):
            # Ones count how many actual input positions overlap the padded kernel.
            count = nn.functional.conv2d(torch.ones(1, 1, side, side), torch.ones(1, 1, 7, 7), padding=3)
            self.assertEqual(count[0, 0, row, col].item(), expected)


if __name__ == '__main__':
    unittest.main()
