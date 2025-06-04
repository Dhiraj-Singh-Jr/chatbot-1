from google import genai
from google.genai.types import GenerateContentConfig, Tool
from IPython.display import display, HTML, Markdown
import io
import json
import re

# Load Gemini API key securely from Colab userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# Select Gemini Model (Use 2.5 Pro if available to you)
MODEL_ID = "gemini-2.5-pro-exp-03-25"  # or "gemini-2.5-pro"

# Define the company to research
COMPANY = 'Alphabet'  # You can change this to any company

# ✨ Enhanced System Instruction for extensive report generation
sys_instruction = """
You are a top-tier financial analyst conducting a detailed research report on a major company.

When given a company name, your report must cover the following sections:

1. **Company Overview**: History, founder(s), mission, and vision.
2. **Business Model**: Revenue sources, products, and services.
3. **Financial Performance**: Key financials (revenue, net income, market cap, etc.).
4. **Market Position**: Competitors, market share, and global reach.
5. **Recent News and Developments**: Key updates from the past 6–12 months.
6. **Leadership and Management**: CEO, executive team, and board structure.
7. **SWOT Analysis**: Strengths, Weaknesses, Opportunities, and Threats.
8. **Future Outlook and Strategy**: Growth strategy, innovation, and long-term goals.

Use Google Search to get the most current and grounded data. Structure your output clearly and concisely.

When ready to write the final report, begin the section with a line of dashes (`---`). Only include the report after this line.
"""

# Configure generation with search tool
config = GenerateContentConfig(
    system_instruction=sys_instruction,
    tools=[Tool(google_search={})],
    temperature=0.4,
)

# Run generation with streaming output
response_stream = client.models.generate_content_stream(
    model=MODEL_ID,
    config=config,
    contents=[COMPANY]
)

# Capture report
report = io.StringIO()

for chunk in response_stream:
    candidate = chunk.candidates[0]

    for part in candidate.content.parts:
        if part.text:
            display(Markdown(part.text))

            # Extract content after the --- line
            if m := re.search(r'(^|\n)-{3,}\n(.*)$', part.text, re.M | re.S):
                report.write(m.group(2))
            elif report.tell():
                report.write(part.text)
        else:
            print(json.dumps(part.model_dump(exclude_none=True), indent=2))

    # Show Google Search suggestions (if any)
    if gm := candidate.grounding_metadata:
        if sep := gm.search_entry_point:
            display(HTML(sep.rendered_content))
