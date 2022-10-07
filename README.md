# :art: Perfect Prompt
An approach to creating the perfect prompt for any image generation task. Type in a prompt, and
Perfect Prompt will match it to one of five art styles (cyberpunk, steampunk, water colors, cottagecore, 
and photorealistic) and provide generated variations of your prompt along these lines, in
under a minute!

Want to try it for yourself? Follow the instructions 
below or [click here](https://arjunpatel7-perfect-prompt-app-mckes9.streamlitapp.com/
) for the deployed version!

![demo gif](https://github.com/arjunpatel7/perfect-prompt/blob/main/perfect_prompt_shorter.gif)


## Requirements
* cohere
* streamlit
* replicate (for image generation)
* pandas

## Installation
First, secure API keys for both Replicate and Cohere,
and save them to Streamlit secrets management.

Clone this repo, and run the following command:
```bash
streamlit run app.py
```

## Author
This app was made by Arjun Patel, a data scientist with experience
applying deep learning to audio and text data. 
Connect with me on [Linkedin](https://www.linkedin.com/in/arjunkirtipatel/)!

## Acknowledgements
This app was made during the Cohere AI Hackathon, in which
Cohere and Lablab.ai offered access to Cohere Classify and Generate
APIs for novel uses. Thank you to both groups for their assistance and mentorship.

This app makes use of the Lexica API, which is generously made freely accessible
by the folks over at [Lexica.art](https://lexica.art). Finally, thanks to Replicate for an easy to use api for Stable Diffusion access.

