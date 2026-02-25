# Canary Prompts

Run these after every training iteration. Save every output. These are your consistent signal across runs.

## Canary A — Known Topic, Short (all 4 models)

A topic you've already written about, kept under 750 words. Lets you compare model output against your actual essay and hear the gap.

**Prompt:**


**Actual essay for comparison:** (title + link)


## Canary B — Novel Topic, Short (all 4 models)

A topic you haven't written about, kept under 750 words. Tests whether the model learned your voice or just your content.

If A sounds like you but B doesn't -> memorized content, not voice.
If A and B both sound like you -> generalized voice.

**Prompt:**


## Canary C — Known Topic, Long (Llama only)

A topic you've already written about, full essay length. Tests whether Llama learned essay-level architecture — threads that pay off across sections, endings that reshape beginnings.

If A and B sound like you but C doesn't -> learned voice without learning structure.

**Prompt:**


**Actual essay for comparison:** (title + link)
