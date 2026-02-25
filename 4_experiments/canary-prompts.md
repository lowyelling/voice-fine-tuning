# Canary Prompts

Run these after every training iteration. Save every output. These are your consistent signal across runs.

## Canary A — Known Topic, Short (all 4 models)

A topic you've already written about, kept under 750 words. Lets you compare model output against your actual essay and hear the gap.

**Prompt:**

Write a personal Substack Note/Tweet about class in America, told from the perspective of a Chinese first generation immigrant whose family is lower-middle class. 

**Actual essay for comparison:** (title + link)
https://substack.com/@lilyfc/note/c-79908566?r=1mz4jf&utm_source=notes-share-action&utm_medium=web


## Canary B — Novel Topic, Short (all 4 models)

A topic you haven't written about, kept under 750 words. Tests whether the model learned your voice or just your content.

If A sounds like you but B doesn't -> memorized content, not voice.
If A and B both sound like you -> generalized voice.

**Prompt:**

Write a personal Substack Note/Tweet about Eileen Gu and Alyssa Liu, both winter Olympic gold medalists. Both grew up in the Bay Area, are half-asian and half-white, conceived via anonymous egg donor, and raised by a single parent. Eileen competed for China in skiing and is maximizing her influencer career while studying at Stanford while Alyssa competed for the United States, took breaks from skating, and is inactive on social media. 

## Canary C — Known Topic, Long (Llama only)

A topic you've already written about, full essay length. Tests whether Llama learned essay-level architecture — threads that pay off across sections, endings that reshape beginnings.

If A and B sound like you but C doesn't -> learned voice without learning structure.

**Prompt:**

Write an essay about Jacques Ellul and propaganda as a technology of conformity, weaving in his two most famous works The Technological Society and Propaganda to critique 21st century events and trends like the COVID-19 pandemic and social media. Include citations about memetic/mimetic Girardian theory to explain Ellul's ellusive, shadowy influence compared to far more well known contemporaries. 

**Actual essay for comparison:** (title + link)
Title: The Forgotten Prophet of Propaganda
Subtitle: Anti-memetic thinker Jacques Ellul on the modern disease of technological efficiency and conformity
https://furniturecoins.substack.com/p/the-forgotten-prophet-of-propaganda