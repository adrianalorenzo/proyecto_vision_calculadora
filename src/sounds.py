import os

# ---------------------------------------------------------------
# Sound effects and background music (pygame.mixer).
# If pygame is not installed or there is no audio device,
# everything still works, just without sound.
# ---------------------------------------------------------------

sounds_folder = "../assets/sounds"
effect_names = ["click", "card_ok", "error", "locked", "unlock", "correct", "point", "game_over", "record", "reset"]
effects_volume = 0.6
music_volume = 0.25

effects = {}
enabled = False
music_on = False

try:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    for name in effect_names:
        effects[name] = pygame.mixer.Sound(os.path.join(sounds_folder, f"{name}.wav"))
        effects[name].set_volume(effects_volume)
    enabled = True
except Exception as error:
    print("No sound:", error)


def play(name: str):
    if enabled and name in effects:
        effects[name].play()


def start_music():
    # Background loop (it does not restart if it is already playing)
    global music_on
    if not enabled or music_on:
        return
    pygame.mixer.music.load(os.path.join(sounds_folder, "music.wav"))
    pygame.mixer.music.set_volume(music_volume)
    pygame.mixer.music.play(-1)
    music_on = True


def toggle_music():
    global music_on
    if not enabled:
        return
    if music_on:
        pygame.mixer.music.pause()
    elif pygame.mixer.music.get_pos() >= 0:
        pygame.mixer.music.unpause()
    else:
        start_music()
        return
    music_on = not music_on
