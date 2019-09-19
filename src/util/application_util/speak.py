

from google.cloud import texttospeech
import playsound

## 구글 Cloud Text - to - speech Api
class Speak:

    def __init__(self):

        # Instantiates a client
        # Set the text input to be synthesized
        self.client = texttospeech.TextToSpeechClient()

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        self.voice = texttospeech.types.VoiceSelectionParams(
            language_code='ko-KR',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

        # Select the type of audio file you want returned
        self.audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)


    def speak_text(self,input_text):
        synthesis_input = texttospeech.types.SynthesisInput(text=input_text)

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = self.client.synthesize_speech(synthesis_input, self.voice, self.audio_config)

        # The response's audio_content is binary.
        with open('output.mp3', 'wb') as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            print('Audio content written to file "output.mp3"')

        playsound.playsound('output.mp3')