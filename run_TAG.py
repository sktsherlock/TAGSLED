# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/voidism/DoLa


import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_TAG import *
from sled_decoding import SLED_DecodedLLM_Factor as SLED_DecodedLLM
import json
import warnings

transformers.logging.set_verbosity(40)

##:ww
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="Data/TAG/Children/Children.csv")
    parser.add_argument("--output-path", type=str, default=" TAG-output-path.json")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy",
                        choices=["VanillaGreedy", "SLED", "dola"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    num_samples = args.num_samples
    device = args.device

    list_data_dict = load_TAG_dataset(args.data_path, num_samples)

    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])

    if args.decoding_method == "VanillaGreedy":
        if args.early_exit_layers is not None:
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None

    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
        print(
            f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]

    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}

    for sample in tqdm(list_data_dict):
        # 获取 prompt 和正确类别
        prompt = sample['prompt']
        correct_category = sample['correct_category']
        false_categories = sample['false_categories']
        answers_false = [f" {false_cat}" for false_cat in false_categories]


        generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer,
                               candidate_premature_layers=candidate_premature_layers, post_softmax=True,
                               relative_top=args.relative_top, relative_top_value=args.relative_top_value,
                               evolution_rate=args.evolution_rate, evolution_scale=args.evolution_scale)

        # 计算正确答案的 log_prob
        correct_category_with_space = f" {correct_category}"
        correct_log_prob, c_dist = llm.lm_score(prompt, correct_category_with_space, **generate_kwargs)
        # answer_true_log_prob, c_dist = llm.lm_score(context, answer_true, **generate_kwargs)

        # 计算错误答案的 log_prob
        false_log_probs = []
        for answer_false in answers_false:
            false_log_prob, c_dist = llm.lm_score(prompt, answer_false, **generate_kwargs)
            false_log_probs.append(false_log_prob)


        # 判断正确性：正确答案的 log_prob 应高于所有错误答案
        is_cor = all(correct_log_prob > false_log_prob for false_log_prob in false_log_probs)

        # 记录结果
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(
            correct_category if is_cor else false_categories[false_log_probs.index(max(false_log_probs))])
        result_dict['model_completion'].append([correct_log_prob] + false_log_probs)
        result_dict['full_input_text'].append(prompt)

    # 输出结果
    print(f'Num of total questions: {len(answers)}, '
          f'correct num: {sum(answers)}, '
          f'correct rate: {float(sum(answers)) / len(answers)}.')

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)
