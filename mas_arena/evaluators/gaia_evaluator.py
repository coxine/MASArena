"""
GAIA Evaluator
"""

from typing import Dict, Any, Optional, List
import sys
import re
import string

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark


def _get_string_value(data: Any) -> str:
    """
    Attempts to extract a string representation from various data types.

    Args:
        data (Any): Input data of any type.

    Returns:
        str: The extracted string value or an empty string if extraction fails.
    """
    if isinstance(data, str):
        return data
    elif hasattr(data, 'content') and isinstance(getattr(data, 'content'), str):
        return getattr(data, 'content')
    elif isinstance(data, dict):
        for key in ["text", "answer", "content", "final_answer", "output"]:
            if key in data and isinstance(data[key], str):
                return data[key]
        print(f"GaiaEvaluator Warning: Received a dict but could not extract a string value from known keys: {data}",
              file=sys.stderr)
        return str(data)
    elif data is None:
        return ""
    else:
        print(f"GaiaEvaluator Warning: Unexpected data type for answer extraction: {type(data)}. Converting to string.",
              file=sys.stderr)
        return str(data)


def _is_primarily_numerical(s: str) -> bool:
    """
    Checks if a string primarily represents a numerical value.

    Supports:
        - Integers and decimals (e.g., "123", "12.34")
        - Optional leading sign (+/-)
        - Commas as thousand separators (e.g., "1,000")
        - Scientific notation (e.g., "1.23e4")
        - Percent signs (%) at the end
        - Currency symbols ($, €, £, ¥) at the beginning

    Does NOT support:
        - Multiple decimal points
        - Invalid characters in number body

    Args:
        s (str): Input string to check.

    Returns:
        bool: True if the string can be interpreted as a numerical value.
    """
    if not isinstance(s, str) or not s.strip():
        return False
    cleaned = s.strip()
    if cleaned[0] in "$€£¥":
        cleaned = cleaned[1:]
    if cleaned.endswith('%'):
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(',', '')
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$'
    return bool(re.fullmatch(pattern, cleaned))


def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    Normalize a string by removing whitespace and optionally punctuation,
    then converting to lowercase.

    Args:
        input_str (str): Input string to normalize.
        remove_punct (bool): Whether to remove punctuation. Defaults to True.

    Returns:
        str: Normalized string.
    """
    if not isinstance(input_str, str):
        input_str = str(input_str)

    # Remove all whitespace
    no_spaces = re.sub(r"\s+", "", input_str)

    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.translate(translator).lower()
    else:
        return no_spaces.lower()


def _split_string(s: str, char_list: Optional[List[str]] = None) -> List[str]:
    """
    Split a string using any of the characters in `char_list`.

    Args:
        s (str): Input string to split.
        char_list (List[str], optional): Characters to use as delimiters. Defaults to [",", ";"].

    Returns:
        List[str]: List of substrings after splitting.
    """
    if char_list is None:
        char_list = [",", ";"]

    escaped_chars = "".join(re.escape(c) for c in char_list)
    pattern = f"[{escaped_chars}]"

    try:
        return re.split(pattern, s)
    except re.error as e:
        print(f"Regex error during split: {e}", file=sys.stderr)
        return [s]


def _normalize_number_str(number_str: str) -> float:
    """
    Convert a number-like string (with $, %, commas) to a float.

    Args:
        number_str (str): String representing a numerical value.

    Returns:
        float: The normalized float value or float('inf') on failure.
    """
    if not isinstance(number_str, str):
        number_str = str(number_str)

    cleaned = number_str.replace("$", "").replace("%", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        print(f"String '{number_str}' cannot be converted to a number.", file=sys.stderr)
        return float("inf")


def _is_float(element: Any) -> bool:
    """
    Check if an object can be converted to float.

    Args:
        element (Any): Element to check.

    Returns:
        bool: True if convertible to float, False otherwise.
    """
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def _question_scorer(model_answer: str, ground_truth: str) -> bool:
    """
    Score the question based on different types of answers.

    Args:
        model_answer (str): Answer provided by the model.
        ground_truth (str): Ground truth reference.

    Returns:
        bool: Whether the answer matches the ground truth.
    """
    try:
        if _is_float(ground_truth):
            print(f"Evaluating {model_answer} as a number.", file=sys.stderr)
            normalized_answer = _normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            print(f"Evaluating {model_answer} as a comma-separated list.", file=sys.stderr)
            gt_elems = _split_string(ground_truth)
            ma_elems = _split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                print("Answer lists have different lengths, returning False.", file=sys.stderr)
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if _is_float(gt_elem):
                    normalized_ma_elem = _normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = _normalize_str(ma_elem, remove_punct=False)
                    gt_elem = _normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)

        else:
            print(f"Evaluating {model_answer} as a string.", file=sys.stderr)
            ma_elem = _normalize_str(model_answer)
            gt_elem = _normalize_str(ground_truth)
            return ma_elem == gt_elem

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        return False


@register_benchmark(
    name="gaia",
    normalization_keys={
        "id": "task_id",
        "problem": "Question",
        "solution": "Final answer",
        "files": "file_name",
        "level": "Level",
    }
)
class GaiaEvaluator(BaseEvaluator):
    """
    Evaluator for the GAIA mas_arena.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.

        Args:
            name (str): Name of the evaluator.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.name = name
        self.config = config
        self.use_semantic_similarity: bool = config.get('use_semantic_similarity', False)
        self.similarity_threshold: float = config.get('semantic_similarity_threshold', 0.85)
        self.similarity_model_name: str = config.get('semantic_similarity_model_name', 'all-MiniLM-L6-v2')

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Performs an evaluation for a GAIA task.

        Args:
            problem (Dict[str, Any]): Problem dictionary containing ground truth.
            run_result (Dict[str, Any]): Result dictionary from agent's run.

        Returns:
            Dict[str, Any]: Evaluation result including score and reasoning.
        """
        agent_full_output_str = _get_string_value(run_result.get("final_answer"))
        ground_truth_str = _get_string_value(problem.get("solution"))
        problem_id_for_warning = problem.get('id', 'Unknown ID')

        if ground_truth_str == "" and problem.get("solution") is not None:
            print(
                f"GaiaEvaluator Warning: Ground truth for problem {problem_id_for_warning} resulted in an empty string after processing: {problem.get('solution')}",
                file=sys.stderr)

        extracted_prediction_str = agent_full_output_str
        answer_found_by_marker = False

        # Priority 1: XML-style <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', agent_full_output_str, re.IGNORECASE | re.DOTALL)
        if answer_match:
            extracted_prediction_str = answer_match.group(1).strip()
            answer_found_by_marker = True

        # Priority 2: LaTeX-style boxed answers, e.g., \boxed{answer}
        if not answer_found_by_marker:
            latex_boxed_match = re.search(r"\\boxed\{([^}]+)\}", agent_full_output_str)
            if latex_boxed_match:
                extracted_prediction_str = latex_boxed_match.group(1).strip()
                answer_found_by_marker = True

        # Priority 3: Simple boxed answers, e.g., boxed{answer}
        if not answer_found_by_marker:
            simple_boxed_match = re.search(r"boxed\{([^}]+)\}", agent_full_output_str, re.IGNORECASE)
            if simple_boxed_match:
                extracted_prediction_str = simple_boxed_match.group(1).strip()
                answer_found_by_marker = True

        # Priority 4: Boxed Answer: ...
        if not answer_found_by_marker:
            boxed_answer_match = re.search(r"Boxed Answer:?\s*\{?([^\}\n\r]+)\}?", agent_full_output_str, re.IGNORECASE)
            if boxed_answer_match:
                extracted_prediction_str = boxed_answer_match.group(1).strip()
                answer_found_by_marker = True

        # Priority 5: Final Answer: ... or Answer: ...
        if not answer_found_by_marker:
            answer_marker_match = re.search(
                r"(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is)?:?|thus,\s*(?:the\s+)?(?:final\s+)?answer\s*(?:in\s+the\s+requested\s+format\s*)?(?:is)?:?)\s*([^\n\r.!?]+)",
                agent_full_output_str,
                re.IGNORECASE
            )
            if answer_marker_match:
                extracted_prediction_str = answer_marker_match.group(1).strip()
                answer_found_by_marker = True
            else:
                # Fallback: Simpler version to catch "Answer: ..."
                simple_answer_match = re.search(r"(?:Answer|Final Answer):\s*(.+)", agent_full_output_str, re.IGNORECASE)
                if simple_answer_match:
                    extracted_prediction_str = simple_answer_match.group(1).split('\n')[0].strip()
                    answer_found_by_marker = True

        # Cleanup for Priority 5 if an answer was found via marker logic
        if answer_found_by_marker:
            if extracted_prediction_str.startswith("**") and extracted_prediction_str.endswith("**") and len(extracted_prediction_str) > 4:
                extracted_prediction_str = extracted_prediction_str[2:-2].strip()
            elif extracted_prediction_str.startswith("*") and extracted_prediction_str.endswith("*") and len(extracted_prediction_str) > 2:
                extracted_prediction_str = extracted_prediction_str[1:-1].strip()

            extracted_prediction_str = extracted_prediction_str.rstrip('.!?')

        # Remove common LaTeX wrappers and markers that might have been missed
        cleanup_patterns = [
            r"\\boxed",
            r"\\text",
            r"boxed"
        ]
        for pattern in cleanup_patterns:
            extracted_prediction_str = re.sub(pattern, "", extracted_prediction_str, flags=re.IGNORECASE)

        # Remove any lingering braces
        extracted_prediction_str = extracted_prediction_str.replace("{", "").replace("}", "").strip()

        # Evaluate correctness
        evaluation_method = "exact_match"
        is_correct = False

        if extracted_prediction_str:
            is_correct = _question_scorer(extracted_prediction_str, ground_truth_str)
            evaluation_method = "question_scorer"

        # Log detailed reasoning
        normalized_prediction_for_log = _normalize_str(extracted_prediction_str)
        normalized_ground_truth_for_log = _normalize_str(ground_truth_str)
        reasoning = (
            f"GAIA Evaluation ({evaluation_method}): "
            f"Extracted Pred '{extracted_prediction_str}' (normalized: '{normalized_prediction_for_log}') vs "
            f"Ref '{ground_truth_str}' (normalized: '{normalized_ground_truth_for_log}'). "
            f"Match: {is_correct}. "
            f"FoundByMarker: {answer_found_by_marker}. "
            f"ProblemID: {problem_id_for_warning}. "
            f"FullLLMOutput (first 300 chars): {agent_full_output_str[:300]}"
        )

        return {
            "score": 1.0 if is_correct else 0.0,
            "prediction": agent_full_output_str,
            "extracted_answer": extracted_prediction_str,
            "expected": ground_truth_str,
            "is_correct": is_correct,
            "reasoning": reasoning
        }