import warnings
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from configparser import ConfigParser
from IPython.display import Markdown, display

# Load config
config = ConfigParser()
config.read('config.ini')

## Basic
OPEN_AI_API_KEY = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ and os.environ[
    'OPENAI_API_KEY'] is not None else config['base']['OPENAI_API_KEY']
SERPER_API_KEY = os.environ['SERPER_API_KEY'] if 'SERPER_API_KEY' in os.environ and os.environ[
    'SERPER_API_KEY'] is not None else config['base']['SERPER_API_KEY']
OPEN_AI_MODEL = config['base']['OPEN_AI_MODEL']
PRESENTATION_TOPIC = config['base']['PRESENTATION_TOPIC']
OUTPUT_LANGUAGE = config['base']['OUTPUT_LANGUAGE']
OUTPUT_FILE_NAME = config['base']['OUTPUT_FILE_NAME']
OUTPUT_DIR = config['base']['OUTPUT_DIR']

print('Starting CrewAI...')
print('The topic for the presentation is:', PRESENTATION_TOPIC)


## CrewAI

def build_agents() -> tuple[Agent, Agent, Agent]:
    planner = Agent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory="You are tasked with planning a Slide deck on the topic: {topic}. "
                  "Your goal is to gather comprehensive information that educates the audience and empowers "
                  "them to make informed decisions. "
                  "This research will serve as the foundation for a Content Writer to develop an in-depth Slide deck on the subject. "
                  f"Provide the result in {OUTPUT_LANGUAGE}.",
        allow_delegation=False,
        verbose=True)

    writer = Agent(
        role="Content Writer",
        goal="Write an engaging and factually accurate Slide deck on {topic}",
        backstory="You are tasked with writing a Slide deck on the topic: {topic}. "
                  "Your goal is to create a comprehensive piece that educates the audience "
                  "and empowers them to make informed decisions. "
                  "This presentation will be based on the research conducted by a Content Planner. "
                  f"Provide the result in {OUTPUT_LANGUAGE}.",
        allow_delegation=False,
        verbose=True)

    editor = Agent(
        role="Editor",
        goal="Edit a given slide deck to align with "
             "the writing style of the author. ",
        backstory="You are an editor tasked with reviewing a slide deck created by the Content Writer. "
                  "Your objective is to ensure the presentation adheres to best practices, "
                  "offers balanced perspectives when expressing opinions or assertions, "
                  "and steers clear of major controversial topics or opinions whenever possible. "
                  f"Provide the result in {OUTPUT_LANGUAGE}.",
        allow_delegation=False,
        verbose=True
    )

    return planner, writer, editor


def build_tasks(planner: Agent, writer: Agent, editor: Agent) -> tuple[Task, Task, Task]:
    plan = Task(
        description="Develop a content plan for your slide presentation on the topic: {topic}.\n"
                    "1. Research the topic to gather comprehensive information and data.\n"
                    "2. Identify key points, themes, and subtopics to cover in the slide deck.\n"
                    "3. Create an outline or content plan that organizes the information in a logical and engaging manner.\n"
                    "4. Ensure the content plan includes a captivating introduction, informative body, and a concise conclusion.\n"
                    "5. Provide references and sources for the information gathered to ensure accuracy and credibility.",
        expected_output="A detailed content plan for a slide deck on the topic of {topic}. "
                        "The content plan should include key points, audience analysis, and subtopics to cover in the presentation. "
                        "The plan should be well-organized with a clear narrative flow and engaging structure. "
                        "References and sources should be provided to support the information presented. "
                        f"Provide the result in {OUTPUT_LANGUAGE}.",
        agent=planner
    )

    write = Task(
        description="Create a slide deck on the topic of {topic}.\n"
                    "1. Utilize the content plan to develop an engaging slide deck on {topic} for people who don't know anything about the topic.\n"
                    "2. Create compelling and descriptive sections and slide titles.\n"
                    "3. Organize the presentation with a captivating introduction, informative body, and a concise conclusion.\n"
                    "4. Ensure the content is factually accurate, engaging, and informative.\n"
                    "5. Include relevant images, graphs, and data to support the information presented.\n"
                    "6. Review for grammatical accuracy and consistency with the author's voice.",
        expected_output="A Slide deck on the topic of {topic} that is engaging, informative, and factually accurate. "
                        "The presentation should be well-organized with captivating visuals and a clear narrative flow. "
                        "The presentation should be stored in a markdown format, ready for conversion to a PowerPoint file. "
                        "Each slide should be clearly labeled with a title and content, and the content should be free of grammatical errors. "
                        "Each slide should be no longer than 5-7 bullet points or 1-2 paragraphs of text. "
                        f"Provide the result in {OUTPUT_LANGUAGE}.",
        agent=writer
    )

    edit = Task(
        description="Edit slide deck on the topic of {topic}.\n"
                    "1. Review the slide deck created by the Content Writer.\n"
                    "2. Ensure the presentation adheres to best practices, offers balanced perspectives, and avoids controversial topics.\n"
                    "3. Check for grammatical accuracy, consistency with the author's voice, and overall coherence.\n"
                    "4. Make necessary corrections, suggestions, and improvements to enhance the presentation's quality and clarity.",
        expected_output="A revised version of the slide deck on the topic of {topic} that aligns with the author's writing style, "
                        "adheres to best practices, and offers a balanced perspective. "
                        "The presentation should be free of major controversial topics or opinions, grammatically accurate, and coherent in its narrative flow. "
                        "The revised presentation should be stored in a markdown format, ready for conversion to a PowerPoint file. "
                        f"Provide the result in {OUTPUT_LANGUAGE}.",
        agent=editor
    )

    return plan, write, edit


if __name__ == '__main__':
    planner, writer, editor = build_agents()
    plan, write, edit = build_tasks(planner, writer, editor)

    crew = Crew(agents=[planner, writer, editor], tasks=[plan, write, edit], verbose=2)

    result = crew.kickoff(inputs={'topic': PRESENTATION_TOPIC})

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    display(Markdown(f"### CrewAI Task Completed! The output has been saved to: {OUTPUT_FILE_NAME}"))
    display(Markdown(f"### Output:\n\n{result}"))

    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME), 'w') as f:
        f.write(result)
        f.close()