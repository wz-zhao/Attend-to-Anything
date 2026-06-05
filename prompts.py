IMAGE_PROMPTS = {
    "UI": (
        "Static user interface screenshot showing webpages or mobile apps, with menus, buttons, icons and dense small text; "
        "free-viewing eye-tracking where attention is strongly biased to the top-left, typical of browsing UI layouts and posters."
    ),
    "SalEC": (
        "Static e-commerce product image with packaging, brand logos, price tags and dense short text blocks; "
        "free-viewing eye-tracking dominated by text and logo-driven attention over retail and shopping items."
    ),
    "OSIE": (
        "Static everyday indoor or outdoor scene containing multiple interacting objects and rich semantic relationships; "
        "free-viewing eye-tracking with moderate center bias, commonly attracted to faces, gaze direction, text and object interactions."
    ),
    "Salicon": (
        "Static natural image from a large-scale object-in-context dataset, covering diverse everyday scenes and environments; "
        "free-viewing saliency data with weak to moderate center bias, designed for general-purpose saliency learning."
    ),
    "CAT2000": (
        "Static high-resolution image from a wide variety of categories including natural scenes and artificial patterns such as cartoons, sketches, fractals and abstract textures; "
        "free-viewing saliency with strong variability in image style and category-dependent center bias."
    ),
    "MIT1003": (
        "Static natural photograph with strong photographer-style center framing; "
        "free-viewing eye-tracking strongly attracted to faces, people, readable text, animals and vehicles, exhibiting a strong center bias."
    ),
    "fiwi": (
        "Static webpage screenshot containing mixed visual elements such as text blocks, images, icons and faces in structured page layouts; "
        "free-viewing eye-tracking where attention is strongly guided by text regions, faces and layout-driven positional bias typical of webpage browsing."
    ),
    "None": (
        "Generic visual stimulus observed under free-viewing conditions. "
        "Represents general human visual attention without assumptions about scene type, task or domain."
    ),
}


VIDEO_PROMPTS = {
    "DHF1K": (
        "Dynamic free-viewing video across diverse scenes and camera motions, containing multiple moving objects and complex backgrounds; "
        "saliency-style eye-tracking with dispersed attention and weak center bias."
    ),
    "Hollywood": (
        "Dynamic cinematic movie clip with actors, dialogues, shot changes and film-style camera motion; "
        "task-driven action recognition with strong center framing and typical movie cinematography."
    ),
    "UCF": (
        "Dynamic broadcast sports video showing athletes on courts or fields, often with uniforms, scoreboards and audience context; "
        "task-driven action recognition with strong center tracking of the main athlete and sports action."
    ),
    "DIEM": (
        "Audio-visual real-world and cinematic video clips including film, television, online video and everyday events; "
        "free-viewing eye-tracking capturing how gaze behavior shapes visual perception, memory and emotional experience over time."
    ),
    "AVAD": (
        "Audio-visual video clips centered on moving sound-generating objects, such as speaking faces, musical performances and action events; "
        "free-viewing eye-tracking emphasizing how tightly coupled audio and motion cues jointly guide visual attention over time."
    ),
    "Coutrot_db1": (
        "Audio-visual conversational video clips featuring multiple interacting people in realistic social environments; "
        "free-viewing eye-tracking revealing attention toward talking faces over time."
    ),
    "Coutrot_db2": (
        "Audio-visual video clips spanning landscapes, moving objects, and social conversations; "
        "free-viewing eye-tracking capturing how auditory cues modulate attention toward sound sources and talking faces."
    ),
    "ETMD_av": (
        "Audio-visual video cinematic clips from Oscar-winning films featuring high-motion action, dialogues, and complex visual storytelling; "
        "free-viewing eye-tracking showing attention guided by chromatic and semantic cues such as faces over long durations."
    ),
    "SumMe": (
        "Audio-visual video clips depicting events like sports and holidays with static, moving, and egocentric camera motions; "
        "task-driven video summarization capturing human consensus on interestingness via explicit selection of important segments."
    ),
    "None": (
        "Generic visual stimulus observed under free-viewing conditions. "
        "Represents general human visual attention without assumptions about scene type, task or domain."
    ),
}


def get_default_prompt_texts():
    prompt_texts = {}
    prompt_texts.update(IMAGE_PROMPTS)
    prompt_texts.update(VIDEO_PROMPTS)
    return prompt_texts
