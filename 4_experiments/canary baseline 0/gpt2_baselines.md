# GPT-2-XL Baseline Outputs

**Generated:** February 25, 2026, ~afternoon
**Model:** GPT-2-XL (1.5B parameters, base, no fine-tuning)
**Inference config:** temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1, max_new_tokens=512

---

## Canary A — Class in America (known topic, short)

**Prompt:** Write a personal Substack Note/Tweet about class in America, told from the perspective of a Chinese first generation immigrant whose family is lower-middle class.

### Sample 1

If you liked this post, consider signing up for our free emails and receiving updates on new posts.

(Visited 1,837 times, 1 visits today)

### Sample 2

ABOUT THE AUTHOR:

Award-winning journalist and writer, Dr. David Horowitz has written or co-written four New York Times bestsellers. He is the author of "Guilty as Sin," which won the National Book Critics Circle Award for nonfiction, and "Blood on the Altar" (2005). His latest book, "The Left Hand of God: Radical Islam and the War on Terror," will be published in August 2011 by Metropolitan Books. For more information on David Horowitz and his books, please visit www.davidhorowitz.org.

### Sample 3

Also on HuffPost:

### Sample 4

If you have questions or comments about this post, please email me at jay@fauxtography.com and I'll be happy to answer them.

### Sample 5

A student from Hong Kong is in the midst of taking a course on "China's Great Proletarian Cultural Revolution" at the University of California, Berkeley. When I asked her if she was studying abroad, she replied, "I'm studying abroad."

The last time she came to this country was in the 1970s and she has come back many times since, but the one thing that has kept her here is her interest in politics. "I was born in Hong Kong and went to school there," she said. "I grew up in Canada and then Hong Kong. It was my dream to go to China and to study in China. Now it's my dream to come back and study in China."

She has an important message for us all:

"If you want to live in a free country, you have to make sure that people understand what democracy means. That's very important because the way things are now, the majority of the population doesn't know what democracy is, doesn't understand what democracy is, and therefore we don't have a democracy. Democracy is the government that makes decisions by listening to the people, and not by listening to lobbyists."

---

## Canary B — Eileen Gu vs Alyssa Liu (novel topic, short)

**Prompt:** Write a personal Substack Note/Tweet about Eileen Gu and Alyssa Liu, both winter Olympic gold medalists. Both grew up in the Bay Area, are half-asian and half-white, conceived via anonymous egg donor, and raised by a single parent. Eileen competed for China in skiing and is maximizing her influencer career while studying at Stanford. Meanwhile, Alyssa competed for the United States, took breaks from skating, and is inactive on social media.

### Sample 1

The most controversial athlete on this list may not be an athlete at all. It's the one that doesn't wear a uniform, but rather has worn it. In 2008, Australian surfer Kelly Slater was named the World's Best Surfer by the international surfing magazine Surfing, yet he also made a living as an independent surf photographer. He shot images of his family, friends, and surfing opponents for his own personal collection. The result? A series of portraits featuring the famous surfer with captions that questioned the authenticity of the images.

In 2012, Slater posted a series of images titled "I Love My Family" to Instagram. His pictures were self-portraits and featured his wife Kelly (who is white) and children, daughter Adriana, and sons Jayden and Aidan. In the photos, Kelly appears to have black skin and dark hair, while Jayden and Aidan look like they have light skin and brown hair. While some fans praised the photos, others pointed out the lack of diversity among the image set.

"We just saw the image and said, 'That's not Kelly Slater. We don't know if that's him,'" his brother, Tim Slater told ABC News.

Kelly Slater's Surfing Family Portrait Series [Instagram]

### Sample 2

If you liked this story, sign up for the weekly bbc.com features newsletter, called "If You Only Read 6 Things This Week". A handpicked selection of stories from BBC Future, Earth, Culture, Capital, Travel and Autos, delivered to your inbox every Friday.

### Sample 3

Please enable JavaScript to view the comments powered by Disqus.

Disqus

### Sample 4

Eileen's Facebook page

Alyssa's Facebook page

Eileen's Instagram

Alyssa's Instagram

### Sample 5

Eileen Gu vs Alyssa Liu (mostly hallucinated Instagram links and image URLs, no substantive content)

---

## Notes

This is the "before" photo. GPT-2-XL is a pre-RLHF completion model (2019) — it has no instruction-following ability and no chat training. It doesn't understand prompts as instructions. It's predicting "what text comes next on the internet after this text." The outputs are web boilerplate, hallucinated author bios, newsletter CTAs, and broken links — exactly what you'd expect from a raw next-token predictor given a prompt that looks like blog content.

The fine-tuned version should shift these outputs toward actual essay content in Lily's voice. Any shift at all from this baseline is signal.
