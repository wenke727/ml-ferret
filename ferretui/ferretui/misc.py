

import re
import pandas as pd


def convert_widget_listing_to_dataframe(text, width=None, height=None):
    # Remove the leading text
    text = text.strip().replace('UI widgets present in this screen include ', '')

    # Split the text into entries based on the pattern ']],'
    entries = text.split(']],')

    # Initialize a list to hold the structured data
    data = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # Ensure each entry ends with ']]'
        if not entry.endswith(']]'):
            entry += ']]'

        # Extract the context (text inside quotes)
        context_match = re.search(r'"(.*?)"', entry)
        context = context_match.group(1) if context_match else ''

        # Extract the bounding box (text inside double square brackets)
        bbox_match = re.search(r'\[\[(.*?)\]\]', entry)
        bbox = '[' + bbox_match.group(1) + ']' if bbox_match else ''

        # Remove the context and bbox from the entry to isolate the label
        entry_remainder = entry.replace(f'"{context}"', '').replace(f'[[{bbox_match.group(1)}]]', '').strip()

        # Extract the label (either before or after the context)
        label_match = re.search(r'(Text displaying|Button)', entry_remainder)
        label = label_match.group(1) if label_match else ''

        # Append the extracted information to the data list
        data.append({'label': label, 'context': context, 'bbox': bbox})

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    df['bbox'] = df['bbox'].apply(lambda x: [i / 1000 for i in eval(x)])
    if width and height:
        df['bbox'] = df['bbox'].apply(lambda x: [x[0] * width, x[1] * height, x[2] * width, x[3] * height])

    return df

if __name__ == '__main__':
    # Your unstructured text from Ferret UI
    text = '''
    UI widgets present in this screen include Text displaying "< Apple" [[0, 59, 199, 106]], "Reminders, Don't forget. Use Reminders" Button [[0, 106, 998, 247]], Text displaying "200K RATINGS" [[9, 272, 303, 315]], Text displaying "AGE" [[371, 272, 526, 315]], Text displaying "CATEGORY" [[601, 272, 824, 315]], Text displaying "DEVEI" [[864, 272, 998, 315]], Text displaying "41" [[19, 315, 130, 353]], Text displaying "4+, Years Old" [[321, 315, 544, 378]], Text displaying "Productivity" [[594, 340, 824, 380]], Text displaying "App" [[864, 340, 975, 380]], Text displaying "9:41" [[88, 401, 199, 438]], Text displaying "9:41" [[562, 401, 677, 438]], Text displaying "Lists" [[9, 428, 136, 467]], Text displaying "Today" [[9, 455, 189, 500]], Text displaying "Q Search" [[617, 450, 818, 489]], Text displaying "Morning, Feed the, Reminders - 9:00 AM Daily" [[9, 500, 371, 560]], Text displaying "Reminders - 8:00 AM Weekly" [[9, 560, 371, 587]], Text displaying "Send out team's weekly progress, Reminders - 10:00 AM/Weekly" [[9, 587, 516, 616]], Text displaying "Reminders - 10:00 AM/Weekly" [[9, 616, 371, 635]], Text displaying "Feed" [[9, 635, 130, 662]], Text displaying "Work" [[371, 616, 469, 635]], Text displaying "Afternoon, Reminders - 4:00 PM/Work" [[9, 632, 371, 712]], Text displaying "Reminders - 4:00 PM/Work" [[9, 712, 371, 739]], Text displaying "Pick up soil for succulents, Reminders - 5:00 PM/ Gardening in Home" [[9, 739, 516, 828]], Text displaying "Reminders - 6:00 PM/ Gardening in Home" [[9, 828, 516, 847]], Text displaying "Groceries" [[670, 736, 832, 771]], Text displaying "Home, Personal" [[670, 771, 881, 814]], Text displaying "Travel" [[670, 814, 793, 845]], Text displaying "Tags" [[670, 845, 781, 876]], Text displaying "Anyt" [[670, 876, 781, 906]], Text displaying "Today" [[884, 918, 998, 960]], Text displaying "Games" [[85, 935, 309, 972]], Text displaying "Apps" [[309, 935, 438, 972]], Text displaying "Acade" [[437, 972, 564, 999]], Text displaying "Search" [[742, 935, 891, 972]].
    '''

    # Convert the text to structured data
    df = convert_widget_listing_to_dataframe(text)

    # Display the DataFrame
    df

