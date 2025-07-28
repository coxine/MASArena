"""
Format Prompts Registry

This module provides a registry of format prompts for different benchmarks and datasets.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum, auto


class DatasetType(Enum):
    """Enum for different types of datasets/benchmarks."""
    BBH = auto()
    IFEVAL = auto()
    DROP = auto()
    MMLU = auto()
    MBPP = auto()
    HUMANEVAL = auto()
    MATH = auto()
    CODE = auto()  # Generic code generation format
    GAIA = auto()  # Placeholder for GAIA benchmark


@dataclass
class FormatPrompt:
    """Class to store format prompt information."""
    name: str
    prompt: str
    description: str
    dataset_type: DatasetType


# Format Prompts Registry
FORMAT_PROMPTS: Dict[str, FormatPrompt] = {
    "bbh": FormatPrompt(
        name="BBH",
        prompt="""
- Ensure the final answer is a single line with no extra whitespace or formatting.
- Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - date_understanding: '(A)', '(B)', '(C)', etc.
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like `)`, `]`, `}`, or `>`
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Sports understanding problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'

<answer>
[Your final answer here]
</answer>
""",
        description="Format prompt for Big-Bench Hard (BBH) problems",
        dataset_type=DatasetType.BBH
    ),
    
    "ifeval": FormatPrompt(
        name="IFEVAL",
        prompt="""
- Follow all punctuation, formatting, length, highlighting and stylistic constraints exactly.
- If an instruction forbids an element (e.g. no commas), *never* include it.
- If an instruction sets a minimum (e.g. ≥ 300 words, ≥ 3 highlighted sections), be sure to exceed it.
- Return **only** the finished response text – no explanations, no markdown fences, no extra whitespace.

Begin now. Remember: output only the compliant answer.
""",
        description="Format prompt for IFEVAL benchmark",
        dataset_type=DatasetType.IFEVAL
    ),
    
    "drop": FormatPrompt(
        name="DROP",
        prompt="""
- Your reply **MUST** contain **exactly two** XML-like blocks and nothing else:

   <answer>
   …ONLY the final answer here (no extra words, no units unless they are part of the answer)…
   </answer>

- Remember:
   1. Put **all** reasoning strictly inside <think> … </think>.
   2. The <answer> block must contain only the short answer string required by the question,
      trimmed of leading/trailing spaces.
   3. Output absolutely nothing outside those two blocks.
""",
        description="Format prompt for DROP benchmark",
        dataset_type=DatasetType.DROP
    ),
    
    "mmlu": FormatPrompt(
        name="MMLU",
        prompt="""
- Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem.
- Ensure the final answer is a single line with no extra whitespace or formatting.
- Only output the answer directly to the questions' options, no other text or explanations.
   <answer>
   [Your final answer here, only alphabet letters, e.g. A, B, C, D, no other text or explanations]
   </answer>
""",
        description="Format prompt for MMLU benchmark",
        dataset_type=DatasetType.MMLU
    ),
    
    "code": FormatPrompt(
        name="CODE",
        prompt="""
- Ensure the code follows these formatting requirements:
  1. Use markdown code blocks with python syntax highlighting
  2. Include clear section headers in markdown format
  3. Wrap final answers in <answer> tags
  4. Structure the response with these sections:
     - Implementation Details
     - Features Implemented
     - Optimizations
     - Validated Code (in markdown code block)

- For code validation:
  1. Test against all provided test cases
  2. Verify function signature matches requirements
  3. Check edge cases and constraints
  4. Ensure code is properly formatted and documented

- For code output:
  1. Include docstrings and type hints
  2. Follow PEP 8 style guidelines
  3. Provide clear explanations of implementation
  4. List any optimizations or improvements made

<answer>
## Implementation Details
{Implementation explanation}

## Features Implemented
{List of implemented features}

## Optimizations
{List of optimizations or "None"}

## Validated Code
```python
{Final validated Python code}
```
</answer>
""",
        description="Format prompt for code generation tasks",
        dataset_type=DatasetType.CODE
    ),
    
    "math": FormatPrompt(
        name="MATH",
        prompt=f"""
- Check for any calculation errors or logical flaws
- Put the final answer in the format: \\boxed{{answer}} without any other text inside the box. The final answer directly answers the question.
""",
        description="Format prompt for math problems",
        dataset_type=DatasetType.MATH
    ),
    "gaia": FormatPrompt(
        name="gaia",
        prompt=f"""You are an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, or web browsing, you can handle it all.
Please note that the task may be complex. Do not attempt to solve it all at once. You should break the task down and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.
Please utilize appropriate tools for the task, analyze the results obtained from these tools, and provide your reasoning. Always use available tools such as browser, calcutor, etc. to verify correctness rather than relying on your internal knowledge.
If you believe the problem has been solved, please output the `final answer`. The `final answer` should be given in <answer></answer> format, while your other thought process should be output in <think></think> tags.
Your `final answer` should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
...
""",
        description="Format prompt for GAIA benchmark",
        dataset_type=DatasetType.GAIA
    )
}


def get_format_prompt(dataset_name: str) -> Optional[str]:
    """
    Get the format prompt for a given dataset.
    
    Args:
        dataset_name: Name of the dataset/benchmark
        
    Returns:
        The format prompt string if found, None otherwise
    """
    # Handle special cases for code generation tasks
    if dataset_name in ["mbpp", "humaneval"]:
        return FORMAT_PROMPTS["code"].prompt
    if dataset_name in ["mmlu_pro", "mmlu"]:
        return FORMAT_PROMPTS["mmlu"].prompt
    # Get prompt for other datasets
    prompt_info = FORMAT_PROMPTS.get(dataset_name.lower())
    return prompt_info.prompt if prompt_info else None


def register_format_prompt(name: str, prompt: str, description: str, dataset_type: DatasetType) -> None:
    """
    Register a new format prompt.
    
    Args:
        name: Name of the format prompt
        prompt: The format prompt string
        description: Description of the format prompt
        dataset_type: Type of dataset this prompt is for
    """
    FORMAT_PROMPTS[name.lower()] = FormatPrompt(name, prompt, description, dataset_type) 