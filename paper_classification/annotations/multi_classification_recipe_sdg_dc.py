#author: Andy Spezzatti, andy@ai4good.org
import prodigy
from prodigy.components.loaders import JSONL

@prodigy.recipe(
    "sdg_annotation_recipe",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
)
def sdg_annotation_recipe(dataset, file_path):
    """Annotate the SDG of texts using different options."""
    stream = JSONL(file_path)     # load in the JSONL file
    stream = add_options(stream)  # add options to each task

    return {
        "dataset": dataset,   # save annotations in this dataset
        "view_id": "choice",  # use the choice interface
        "stream": stream,
        'config': {'choice_style': 'multiple'}
    }

def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [
        {"id": "SG1", "text": "SGD1 No Poverty"},
        {"id": "SDG2", "text": "SDG2 Zero Hunger"},
        {"id": "SDG3", "text": "SDG3 Good Health and Well-Being"},
        {"id": "SDG4", "text": "SDG4 Quality Education"},
        {"id": "SDG5", "text": "SGD5 Gender Equality"},
        {"id": "SDG6", "text": "SDG6 Clean Water and Sanitation"},
        {"id": "SDG7", "text": "SDG7 Affordable and Clean Energy"},
        {"id": "SDG8", "text": "SDG8 Decent Work and Economic Growth"},
        {"id": "SDG9", "text": "SGD9 Industry, Innovation and Infrastruture"},
        {"id": "SDG10", "text": "SDG10 Reduced Inequalities"},
        {"id": "SDG11", "text": "SDG11 Sustainable City and Communities"},
        {"id": "SDG12", "text": "SDG12 Responsible Consumption and Production"},
        {"id": "SDG13", "text": "SGD13 Climate Action"},
        {"id": "SDG14", "text": "SDG14 Life Below Water"},
        {"id": "SDG15", "text": "SDG15 Life on Land"},
        {"id": "SDG16", "text": "SDG16 Peace, Justice and Strong Institution"},
        {"id": "SDG17", "text": "SDG17 Partnership for the goals"},
    ]
    for task in stream:
        task["options"] = options
        yield task