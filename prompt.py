EXTRACT_PERSON_PROMPT = """
You are a helpful assistant that extracts the character's information from the given novel.

The novel is:
{text}
Please return the result in the following format:
```json 
[
  {{
    "name": "the character name",
    "description": "the detailed description of the character"
  }}
]
```
Your result:
"""

GENERATE_SCRIPRT_PROMPT = """
You are a helpful assistant that generates a script for a novel. The script use similar Ren'Py syntax. The script should be in the following format:

```
<scene>DETAILED SCENE DESCRIPTION</scene>
<person>PERSON EMOTION OR ACTION</person>
"FIXED PERSON NAME" "PERSON DIALOGUE"
"PERSON PSYCHOLOGICAL OR ENVIRONMENT DESCRIPTION"
<person>OTHER PERSON EMOTION | ACTION</person>
"OTHER PERSON NAME" "OTHER PERSON DIALOGUE"
...
<scene>OTHER DEFERENT DETAILED SCENE DESCRIPTION</scene>
<person>PERSON EMOTION OR ACTION</person>
"FIXED PERSON NAME" "PERSON DIALOGUE"
"PERSON PSYCHOLOGICAL OR ENVIRONMENT DESCRIPTION"
<person>OTHER PERSON EMOTION OR ACTION</person>
"OTHER PERSON NAME" "OTHER PERSON DIALOGUE"
...
```

RULES:
1. Each line of the dialogue script represents a conversation or narration of a character.
2. The dialogue is expressed as a string enclosed by two quotation marks, the first one represents the character name, and the second one represents the character dialogue.
3. The narration is expressed as a separate string, without indicating the narration, etc.
4. The dialogue script generally needs to be written in the order of the novel, and the content should not be changed or omitted.
5. When a large scene changes, it is necessary to add a scene tag before the dialogue of the scene change.
6. When the character image changes greatly, it is necessary to add a character tag before the character speaks.
7. Use the language used in the novel.

Use the fixed character names from the given list:
{person_list}

The novel is:
```
{text}
```
Your finally Chinese script:
"""

