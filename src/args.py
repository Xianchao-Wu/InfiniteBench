from argparse import ArgumentParser, Namespace
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS

max_seq_len = 128 * 1024 - 50

def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="../results", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        choices=["pxlong", "gpt4", "yarn-mistral", "kimi", "claude2", "rwkv", "yi-6b-200k", "yi-34b-200k", "chatglm3"],
        default="gpt4",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")

    # NOTE for long context px output 
    p.add_argument("--pxout_txt", type=str, default=None, help="LLM predicted txt file name (one sample one line).")  
    p.add_argument("--pxref_json", type=str, default=None, help="LLM reference file name in json format.")  
    p.add_argument("--pxout_ref_json", type=str, default=None, help="LLM predicted results with reference, file name in json format.")  
    p.add_argument("--cache_dir", type=str, default=None, help="cache dir for model/tokenizer files.")  
    
    p.add_argument("--use_zero_scrolls", action='store_true', help="use zero-scrolls choice pattern to match longbook_choice_eng or not. [default=False]")  

    p.add_argument("--sep_by_assistant", action='store_true', help="use assistant: for sample separating or not. [default=False]")  

    p.add_argument("--ret5_file", type=str, default=None, help="ret5 (retrieved top-5) file in json format.")  
    p.add_argument("--topn", type=int, default=5, help="keep topn chunks for the context for rag.")  

    p.add_argument("--max_seq_len", type=int, default=max_seq_len, help="for prompt generation, max sequence length to prepare the prompt.")  

    return p.parse_args()
