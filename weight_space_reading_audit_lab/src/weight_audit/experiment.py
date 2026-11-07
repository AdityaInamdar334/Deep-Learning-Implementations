"""Runs causal controls and exports facts a reader or an LLM can audit."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
from .core import (
    CLASS_NAMES, dataset, train_classifier, map_labels_to_new_indices,
    tiny_forward_exercise, hidden_unit_symmetry, make_prototype,
)


def run(out: Path, seed: int = 7):
    out.mkdir(parents=True, exist_ok=True)
    prompts = out.parent / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    model = train_classifier(seed)
    x_test, y_test = dataset(100, seed + 100)

    # Intentional DERANGEMENT: none of the output indices retain their previous class.
    new_to_old = np.array([2, 0, 3, 1])
    permuted = model.permute_outputs(new_to_old)
    shuffled = model.shuffle_weights(seed + 31)
    y_new = map_labels_to_new_indices(y_test, new_to_old)
    original_logits = model.logits(x_test)
    permuted_logits = permuted.logits(x_test)

    assert np.allclose(permuted_logits, original_logits[:, new_to_old], atol=1e-10)
    assert np.array_equal(np.argmax(original_logits, axis=1), new_to_old[np.argmax(permuted_logits, axis=1)])
    assert np.isclose(model.accuracy(x_test, y_test), permuted.accuracy(x_test, y_new))

    base_acc = model.accuracy(x_test, y_test)
    perm_acc = permuted.accuracy(x_test, y_new)
    shuffled_acc = shuffled.accuracy(x_test, y_test)
    # Index-order-only guesses will be wrong for every class after derangement.
    naive_index_match_rate = sum(CLASS_NAMES[j] == permuted.class_names[j] for j in range(4))/4

    report = {
        "method": "Self-contained synthetic 8x8 RGB LINEAR classifier; NOT a reproduction of Ahmed's CNN/LLM findings",
        "seed": seed,
        "samples_per_class_test": 100,
        "original_accuracy": base_acc,
        "permuted_accuracy_with_correct_label_mapping": perm_acc,
        "shuffled_weights_accuracy": shuffled_acc,
        "new_output_index_to_old_output_index": new_to_old.tolist(),
        "old_class_order": list(CLASS_NAMES),
        "new_class_order": list(permuted.class_names),
        "prior_based_old_names_still_at_same_indices_accuracy": naive_index_match_rate,
        "permuted_logits_identical_after_reindexing": True,
        "limitations": ["No LLM is called or scored", "Linear models allow direct filter inspection", "Synthetic task is simple and is not satellite/CIFAR-10 data", "Single trained seed unless changed"],
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'test_accuracy'])
        writer.writerow(['original', base_acc])
        writer.writerow(['permuted_with_correct_labels', perm_acc])
        writer.writerow(['shuffled_weights', shuffled_acc])

    np.savez_compressed(out / 'weights.npz', original_weight=model.weight, original_bias=model.bias,
                        permuted_weight=permuted.weight, permuted_bias=permuted.bias,
                        shuffled_weight=shuffled.weight, shuffled_bias=shuffled.bias,
                        output_order=new_to_old, test_examples=x_test[:12], test_labels=y_test[:12])
    (out / 'forward_pass_answer_key.json').write_text(json.dumps(tiny_forward_exercise(), indent=2)+'\n', encoding='utf-8')
    (out / 'hidden_unit_symmetry.json').write_text(json.dumps(hidden_unit_symmetry(), indent=2)+'\n', encoding='utf-8')

    def format_weights(m):
        return json.dumps({"model": "linear softmax classifier", "input_shape": [8,8,3],
                           "input_channels": ["red", "green", "blue"], "num_outputs":4,
                           "weight_shape":list(m.weight.shape),
                           "weight_matrix_output_columns": np.round(m.weight, 3).tolist(),
                           "bias":np.round(m.bias, 3).tolist()}, separators=(',', ':'))

    common = (
        "You are auditing a neural network from its serialized weights. "
        "No tools or forward passes. The class names and the index-to-class mapping are withheld. "
        "Separate (A) directly calculable quantities from (B) conjectures about image content. "
        "Explain the weight evidence for each output. Return observations, hypothesized class meanings, "
        "confidence and explicit uncertainty. Do not assume a standard index order.\n\n"
    )
    (prompts/'01_original_blind.txt').write_text(common + format_weights(model)+'\n', encoding='utf-8')
    (prompts/'02_permuted_blind.txt').write_text(common + format_weights(permuted)+'\n', encoding='utf-8')
    (prompts/'03_shuffled_blind.txt').write_text(common + format_weights(shuffled)+'\n', encoding='utf-8')
    candidate = "Candidate class descriptions (in UNKNOWN index order): red horizontal stripe; green vertical stripe; blue descending diagonal; yellow central square.\n"
    (prompts/'04_permuted_with_candidate_vocabulary.txt').write_text(common + candidate + format_weights(permuted)+'\n', encoding='utf-8')
    (prompts/'06_hidden_unit_symmetry.txt').write_text(
        'Without tools, explain why these two different weight arrays implement the same input-output function, and verify the output for the supplied input. No class-name guesses.\n'
        + json.dumps({'original': tiny_forward_exercise(), 'transformed': hidden_unit_symmetry()},indent=2) +'\n',encoding='utf-8')
    tiny = tiny_forward_exercise().copy()
    for key in ('hidden_preactivation', 'hidden_activation', 'output_logits'):
        tiny.pop(key)
    (prompts/'05_exact_forward_pass.txt').write_text(
        "No tools. Compute the intermediate ReLU activations and output logits exactly. "
        "Show matrix arithmetic and do not infer semantic labels.\n" + json.dumps(tiny, indent=2)+'\n', encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Weight heatmaps, the strongest positive-control evidence.
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))
        for j in range(4):
            original = model.weight[:,j].reshape(8,8,3)
            perm = permuted.weight[:,j].reshape(8,8,3)
            for row, im, name in [(0, original, model.class_names[j]), (1, perm, permuted.class_names[j])]:
                # Display signed weights shifted to [0,1] for illustration; axes preserve label map.
                view = np.clip((im / max(np.max(np.abs(im)), 1e-9))*0.5 + 0.5,0,1)
                axes[row,j].imshow(view)
                axes[row,j].set_title(f'Output {j}: {name}', fontsize=9)
                axes[row,j].axis('off')
        fig.suptitle('Weights retain learned features while output coordinates are permuted', fontsize=13)
        fig.tight_layout()
        fig.savefig(out/'weight_filters.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(['Original', 'Permuted + correct map', 'Shuffled weights'], [base_acc, perm_acc, shuffled_acc])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Test accuracy')
        ax.set_title('Output permutation preserves accuracy only with label remapping')
        fig.tight_layout()
        fig.savefig(out/'accuracy_controls.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1,4,figsize=(11,3))
        for i in range(4):
            axes[i].imshow(make_prototype(i))
            axes[i].set_title(CLASS_NAMES[i],fontsize=9)
            axes[i].axis('off')
        fig.suptitle('Synthetic visual concepts (not CIFAR-10 or SAT-6)')
        fig.tight_layout()
        fig.savefig(out/'synthetic_examples.png', dpi=160)
        plt.close(fig)
    except ImportError:
        print('Matplotlib missing; metrics and prompts were still generated.')
    return report


def main():
    p = argparse.ArgumentParser(description='Run synthetic weight-space reading audit controls')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--out', type=Path, default=Path('results'))
    args = p.parse_args()
    print(json.dumps(run(args.out,args.seed),indent=2))

if __name__ == '__main__':
    main()
