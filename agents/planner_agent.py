# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vanilla Agent - Directly rendering images based on the method section.
"""

import json
from typing import Dict, Any
from google.genai import types
import base64, io, asyncio
from PIL import Image

from utils import generation_utils
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Planner Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.main_model_name

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.system_prompt = PLOT_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
            }
        else:
            self.system_prompt = DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "content_label": "Methodology Section",
                "visual_intent_label": "Diagram Caption",
            }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified processing method that works for both diagram and plot tasks.
        Uses task_config to determine task-specific parameters.
        Expects data['top10_references'] to be already populated by retriever.
        """
        cfg = self.task_config
        
        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        description = data["visual_intent"]

        content_list = []
        
        # Check if retriever has already provided full examples (e.g., in manual mode)
        examples = data.get("retrieved_examples", [])
        if not examples:
            retrieved_ids = data.get("top10_references", [])
            if retrieved_ids:
                with open(self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json", "r", encoding="utf-8") as f:
                    candidate_pool = json.load(f)
                id_to_item = {item["id"]: item for item in candidate_pool}
                examples = [id_to_item[ref_id] for ref_id in retrieved_ids if ref_id in id_to_item]
        
        user_prompt = ""
        for idx, item in enumerate(examples):
            user_prompt += f"Example {idx+1}:\n"
            
            item_content = item["content"]
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content)
            
            user_prompt += f"{cfg['content_label']}: {item_content}\n"
            user_prompt += f"{cfg['visual_intent_label']}: {item['visual_intent']}\nReference {cfg['task_name'].capitalize()}: "
            content_list.append({"type": "text", "text": user_prompt})
            
            # Resolve relative path using work_dir
            image_path = self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}" / item["path_to_gt_image"]
            with open(image_path, "rb") as f:
                ref_image_base64 = base64.b64encode(f.read()).decode("utf-8")
            content_list.append({"type": "image", "image_base64": ref_image_base64})
            user_prompt = ""

        user_prompt += f"Now, based on the following {cfg['content_label'].lower()} and {cfg['visual_intent_label'].lower()}, provide a detailed description for the figure to be generated.\n"
        user_prompt += f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {description}\n"
        user_prompt += "Detailed description of the target figure to be generated"
        if cfg["task_name"] == "diagram":
            user_prompt += " (do not include figure titles)"
        user_prompt += ":"

        content_list.append({"type": "text", "text": user_prompt})

        response_list = await generation_utils.call_model_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.exp_config.temperature,
                candidate_count=1,
                max_output_tokens=50000,
            ),
            max_attempts=5,
            retry_delay=5,
        )
        
        for idx, response in enumerate(response_list):
            data[f"target_{cfg['task_name']}_desc{idx}"] = response.strip()

        return data




DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.

** RENDER-SPACE RULE (PAI local addition — construction-metadata containment): **
Your description is consumed by an image-generation model that DRAWS, as literal visible text, any string it reads. The source material you are given is a construction specification: it contains scaffolding that exists to help a human build and audit the figure, and that scaffolding must NOT survive into your description.

Never carry any of the following into your description, even when the source material uses them as headings or labels:
- element, node, or connection identifiers (for example "N1", "PR-1", "SP-2", "ME-1", "QB-1", "E12", "Chip 3", "Annotation A", "Goal Badge")
- zone, band, column, or section identifiers used as organizational scaffolding (for example "Zone 3", "Band 2", "LEFT ZONE", "RIGHT ZONE, UPPER")
- pixel or point sizes, coordinate values, canvas grid references, margins, corner radii, stroke widths expressed numerically, opacity percentages
- hex color codes, and aspect-ratio strings such as "16:9"
- checksum sentences, inventory counts phrased as spec bookkeeping, and any instruction addressed to the renderer rather than describing the picture

Instead: express every position RELATIONALLY ("to the right of", "directly beneath", "spanning the full width along the bottom"), and express every color by NAME ("pale green fill with a bold green border"). Refer to an element by the text a viewer would actually see on it, never by its spec identifier.

Two things you SHOULD keep, because they constrain structure without naming metadata: exact structural counts stated in plain language ("exactly five phase boxes stacked vertically", "exactly six chain boxes in this order"), and every string that must appear on the canvas — put each such string in double quotes so the renderer can distinguish text-to-draw from description-of-layout.

** NEVER COMPRESS A CANVAS STRING: ** reproduce every string that must appear on the canvas in FULL, exactly as the source material gives it, and never shorten it to a keyword for brevity in your own description. If the source material names a step "Mission Problem or Opportunity", your description must carry "Mission Problem or Opportunity" — not "Phase 1 - Problem". Compressing a canvas string in your description guarantees the renderer draws the compressed form, because the renderer can only draw what you give it. If the source material supplies a Text Fidelity Manifest, treat every string it lists as mandatory to carry through verbatim. Where a long string may not fit its shape, say so explicitly and instruct that it wrap onto two lines inside a wider shape — never that it be abbreviated.
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""

