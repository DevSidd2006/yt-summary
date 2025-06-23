import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from deep_translator import GoogleTranslator
import re
from urllib.parse import parse_qs, urlparse
import yt_dlp
import tempfile
from transformers import pipeline
import torch
from collections import Counter
import requests

# Try to import whisper with error handling
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except Exception as e:
    st.warning(f"Whisper not available: {e}. Audio transcription will be disabled.")
    WHISPER_AVAILABLE = False
    WhisperModel = None

# Initialize summarization pipeline
@st.cache_resource
def load_summarizer():
    """Load the summarization model"""
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=device
    )
    return summarizer

# Add Hugging Face translation pipeline for Hindi to English
@st.cache_resource
def load_hi_en_translator():
    return pipeline('translation', model='Helsinki-NLP/opus-mt-hi-en')

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    return None

def get_video_info(video_id):
    """Get video information including available subtitles"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return transcript_list
    except:
        return None

def get_transcript(video_id, language_codes=['en', 'hi']):
    """Get transcript in specified languages"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manually created subtitles first
        for transcript in transcript_list:
            if transcript.language_code in language_codes and not transcript.is_generated:
                return transcript.fetch(), transcript.language_code, False
        
        # If no manual subtitles, get auto-generated
        for transcript in transcript_list:
            if transcript.language_code in language_codes and transcript.is_generated:
                return transcript.fetch(), transcript.language_code, True
                
        # If no subtitles in desired languages, get any available
        for transcript in transcript_list:
            if not transcript.is_generated:
                return transcript.fetch(), transcript.language_code, False
                
        for transcript in transcript_list:
            if transcript.is_generated:
                return transcript.fetch(), transcript.language_code, True
                
    except Exception as e:
        return None, None, None

def find_ffmpeg_location():
    """Find FFmpeg installation location"""
    # Common FFmpeg locations to check
    potential_locations = [
        # WinGet installation location
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                    'Microsoft', 'WinGet', 'Packages', 
                    'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
                    'ffmpeg-7.1.1-full_build', 'bin'),
        # Other common locations
        'C:\\ffmpeg\\bin',
        'C:\\Program Files\\ffmpeg\\bin',
        'C:\\Program Files (x86)\\ffmpeg\\bin',
    ]
    
    for location in potential_locations:
        if os.path.exists(os.path.join(location, 'ffmpeg.exe')):
            return location
    
    return None

def download_audio(video_url):
    """Download audio from YouTube video"""
    try:
        # Find FFmpeg location
        ffmpeg_location = find_ffmpeg_location()

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        }
        # Add FFmpeg location if found
        if ffmpeg_location:
            ydl_opts['ffmpeg_location'] = ffmpeg_location

        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts['outtmpl'] = os.path.join(temp_dir, 'audio.%(ext)s')

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                audio_file = os.path.join(temp_dir, 'audio.mp3')

                if os.path.exists(audio_file):
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()

                    # Create a temporary file that won't be automatically deleted
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    temp_audio.write(audio_data)
                    temp_audio.close()

                    return temp_audio.name

    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None

@st.cache_resource
def load_whisper_model():
    """Load Whisper model"""
    if not WHISPER_AVAILABLE:
        return None
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_file, language='en'):
    """Transcribe audio using Faster Whisper"""
    if not WHISPER_AVAILABLE:
        st.error("Whisper is not available. Cannot transcribe audio.")
        return None
    
    try:
        model = load_whisper_model()
        if model is None:
            return None
        
        segments, info = model.transcribe(audio_file, language=language)
        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "
        
        return transcript_text.strip()
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def translate_text(text, target_lang='en'):
    """Translate text to target language, using Hugging Face for Hindi to English, else deep-translator."""
    try:
        if target_lang == 'en':
            # Use Hugging Face for Hindi to English
            hi_en_translator = load_hi_en_translator()
            # Split into smaller chunks for long texts
            max_length = 400
            sentences = re.split(r'(?<=[।.!?])\s+', text)
            chunks = []
            current = ''
            for sent in sentences:
                if len(current) + len(sent) < max_length:
                    current += ' ' + sent
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = sent
            if current.strip():
                chunks.append(current.strip())
            translated_chunks = [hi_en_translator(chunk)[0]['translation_text'] for chunk in chunks if chunk.strip()]
            translated_text = ' '.join(translated_chunks)
            # Check if translation is likely English (basic check)
            if re.search(r'[\u0900-\u097F]', translated_text):
                st.warning("Translation may have failed. The result still contains Hindi characters.")
            return translated_text
        else:
            # Use deep-translator for other languages
            max_length = 4000
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            else:
                chunks = [text]
            translated_chunks = []
            for chunk in chunks:
                result = GoogleTranslator(source='auto', target=target_lang).translate(chunk)
                translated_chunks.append(result)
            return ' '.join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def chunk_text(text, max_chunk_size=1024):
    """Split text into chunks for summarization"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text, summarizer):
    """Summarize text using Hugging Face Transformers"""
    try:
        # Check if text is too short
        if len(text.split()) < 50:
            return text
        
        # Split text into chunks if it's too long
        max_chunk_size = 1024  # BART max input length
        chunks = chunk_text(text, max_chunk_size)
        
        if len(chunks) == 1:
            # Single chunk - summarize directly
            summary = summarizer(
                chunks[0], 
                max_length=150, 
                min_length=30, 
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        else:
            # Multiple chunks - summarize each chunk
            chunk_summaries = []
            
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                chunk_summary = summarizer(
                    chunk, 
                    max_length=150, 
                    min_length=30, 
                    do_sample=False,
                    truncation=True
                )
                chunk_summaries.append(chunk_summary[0]['summary_text'])
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # If combined summary is still too long, summarize again
            if len(combined_summary.split()) > 500:
                final_chunks = chunk_text(combined_summary, max_chunk_size)
                final_summaries = []
                
                for chunk in final_chunks:
                    final_summary = summarizer(
                        chunk, 
                        max_length=150, 
                        min_length=30, 
                        do_sample=False,
                        truncation=True
                    )
                    final_summaries.append(final_summary[0]['summary_text'])
                
                return ' '.join(final_summaries)
            else:
                # Final summarization
                final_summary = summarizer(
                    combined_summary, 
                    max_length=300, 
                    min_length=50, 
                    do_sample=False,
                    truncation=True
                )
                return final_summary[0]['summary_text']
            
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return None

def create_chapter_summaries(text, summarizer, num_chapters=5):
    """Create chapter-wise summaries by dividing content into chapters"""
    try:
        words = text.split()
        total_words = len(words)
        
        if total_words < 100:
            return [{"chapter": 1, "title": "Complete Content", "summary": text}]
        
        # Calculate words per chapter
        words_per_chapter = total_words // num_chapters
        chapters = []
        
        progress_bar = st.progress(0)
        
        for i in range(num_chapters):
            start_idx = i * words_per_chapter
            
            # For the last chapter, include all remaining words
            if i == num_chapters - 1:
                end_idx = total_words
            else:
                end_idx = (i + 1) * words_per_chapter
            
            chapter_text = ' '.join(words[start_idx:end_idx])
            
            # Skip if chapter is too short
            if len(chapter_text.split()) < 20:
                continue
            
            # Generate chapter title from first few words
            chapter_title = ' '.join(chapter_text.split()[:8]) + "..."
            
            # Summarize the chapter
            chapter_summary = summarize_text(chapter_text, summarizer)
            
            if chapter_summary:
                chapters.append({
                    "chapter": i + 1,
                    "title": chapter_title,
                    "summary": chapter_summary,
                    "word_count": len(chapter_text.split())
                })
            
            progress_bar.progress((i + 1) / num_chapters)
        
        progress_bar.empty()
        return chapters
        
    except Exception as e:
        st.error(f"Error creating chapter summaries: {str(e)}")
        return None

def process_video(video_url, summarizer, summary_type="Single Summary", num_chapters=None):
    """Main processing function following the algorithm"""
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None
    
    st.info("Fetching video information...")
    
    # Get transcript
    transcript, language, is_generated = get_transcript(video_id)
    
    transcript_text = None
    translated = False
    
    if transcript:
        # Format transcript
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        
        st.success(f"Found {'auto-generated' if is_generated else 'manual'} subtitles in {language}")
        
        # Translate if not in English
        if language != 'en':
            st.info("Translating to English...")
            transcript_text = translate_text(transcript_text, 'en')
            translated = True
    else:
        st.warning("No subtitles found. Attempting to transcribe audio...")
        
        # Download and transcribe audio
        with st.spinner("Downloading audio..."):
            audio_file = download_audio(video_url)
        
        if audio_file:
            with st.spinner("Transcribing audio (this may take a while)..."):
                transcript_text = transcribe_audio(audio_file)
                
                # Clean up
                try:
                    os.remove(audio_file)
                except:
                    pass
        else:
            st.error("Could not download audio from video")
            return None
    
    if transcript_text:
        # Always generate overall summary
        st.info("Generating overall summary...")
        overall_summary = summarize_text(transcript_text, summarizer)
        
        if summary_type == "Chapter-wise Summary" and num_chapters:
            st.info(f"Generating {num_chapters} chapter summaries...")
            chapters = create_chapter_summaries(transcript_text, summarizer, num_chapters)
            return chapters, transcript_text, "chapters", overall_summary, translated
        else:
            st.info("Generating summary...")
            return overall_summary, transcript_text, "single", overall_summary, translated
    else:
        st.error("Could not get transcript for the video")
        return None, None, None, None, None

def get_video_metadata(video_url):
    """Fetch video title and thumbnail using yt_dlp."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info.get('title', ''), info.get('thumbnail', '')
    except Exception:
        return '', ''

@st.cache_data(show_spinner=False)
def cached_get_transcript(video_id, language_codes=['en', 'hi']):
    return get_transcript(video_id, language_codes)

@st.cache_data(show_spinner=False)
def cached_summarize_text(text, _summarizer, length='medium', style='paragraph'):
    # Adjust parameters based on length/style
    if length == 'short':
        max_length, min_length = 80, 20
        chunk_size = 512
    elif length == 'long':
        max_length, min_length = 300, 80
        chunk_size = 1024
    else:
        max_length, min_length = 150, 40
        chunk_size = 1024
    # Chunk text if too long
    chunks = chunk_text(text, chunk_size)
    summaries = []
    for chunk in chunks:
        summary = _summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        summaries.append(summary)
    full_summary = ' '.join(summaries)
    if style == 'bullets':
        # Convert to bullet points (simple split)
        return '\n'.join(['- ' + s.strip() for s in full_summary.split('.') if s.strip()])
    return full_summary

def extract_keywords(text, top_n=8):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stopwords = set(['this','that','with','from','have','will','your','just','like','they','what','when','where','which','their','about','would','there','could','should','because','these','those','being','been','into','only','some','such','than','then','them','very','also','more','most','much','many','each','other','over','after','before','under','again','while','here','make','made','does','did','doing','has','had','can','may','might','must','shall','who','whom','whose','how','why','our','out','off','for','and','the','are','was','were','you','but','not','all','any','his','her','him','she','himself','herself','it','its','itself','we','us','ours','i','me','my','mine','he','him','himself','on','in','at','by','to','of','as','is','an','or','if','so','no','yes','up','down','a'])
    filtered = [w for w in words if w not in stopwords]
    return [w for w, _ in Counter(filtered).most_common(top_n)]

# Streamlit UI
def main():
    st.set_page_config(page_title="YouTube Video Summarizer", page_icon="📺")
    st.title("🎥 YouTube Video Summarizer")
    st.markdown("This app summarizes YouTube videos by extracting subtitles or transcribing audio.")

    # Video URL input
    video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    # Show video metadata
    if video_url:
        title, thumbnail = get_video_metadata(video_url)
        if title:
            st.markdown(f"**Video Title:** {title}")
        if thumbnail:
            st.image(thumbnail, width=320)

    # Settings in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("Summary Type")
        summary_type = st.radio(
            "Choose summary type:",
            options=["Single Summary", "Chapter-wise Summary"],
            index=0
        )
        st.subheader("Summary Length")
        summary_length = st.select_slider(
            "Choose summary length",
            options=["Short", "Medium", "Long"],
            value="Medium"
        )
        st.subheader("Summary Style")
        summary_style = st.radio(
            "Summary style:",
            options=["Paragraph", "Bullets"],
            index=0
        )
        if summary_type == "Chapter-wise Summary":
            st.subheader("Number of Chapters")
            num_chapters = st.select_slider(
                "Divide content into chapters:",
                options=[3, 5, 10],
                value=5
            )
            st.info(f"Content will be divided into {num_chapters} chapters with individual summaries")
        else:
            num_chapters = None

    # Process button
    if st.button("Summarize Video", type="primary"):
        if video_url:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL")
                return
            with st.spinner("Fetching transcript..."):
                transcript, language, is_generated = cached_get_transcript(video_id)
            if transcript:
                formatter = TextFormatter()
                transcript_text = formatter.format_transcript(transcript)
                st.success(f"Found {'auto-generated' if is_generated else 'manual'} subtitles in {language}")
                st.markdown(f"**Detected Language:** {language}")
                # Translate if not in English
                if language != 'en':
                    st.info("Translating to English...")
                    transcript_text = translate_text(transcript_text, 'en')
            else:
                st.warning("No subtitles found. Attempting to transcribe audio...")
                with st.spinner("Downloading audio..."):
                    audio_file = download_audio(video_url)
                if audio_file:
                    with st.spinner("Transcribing audio (this may take a while)..."):
                        transcript_text = transcribe_audio(audio_file)
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                else:
                    st.error("Could not download audio from video")
                    return
            # Show keywords
            keywords = extract_keywords(transcript_text)
            st.markdown(f"**Top Keywords:** {', '.join(keywords)}")
            # Load summarizer (choose model based on summary_length)
            if summary_length == "Short":
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            else:
                summarizer = load_summarizer()
            # Summarize
            if summary_type == "Chapter-wise Summary" and num_chapters:
                st.info(f"Generating {num_chapters} chapter summaries...")
                chapters = create_chapter_summaries(transcript_text, summarizer, num_chapters)
                st.success(f"Generated {len(chapters)} chapter summaries!")
                st.subheader("📚 Chapter Summaries")
                for i, chapter in enumerate(chapters):
                    st.markdown(f"**Chapter {chapter['chapter']}**")
                    st.write(cached_summarize_text(chapter['summary'], summarizer, summary_length.lower(), summary_style.lower()))
                    st.caption(f"Words in this chapter: {chapter['word_count']}")
            else:
                st.info("Generating summary...")
                summary = cached_summarize_text(transcript_text, summarizer, summary_length.lower(), summary_style.lower())
                st.success("Summary generated successfully!")
                st.subheader("📝 Summary")
                st.write(summary)
            # Download transcript
            st.download_button(
                label="📥 Download Transcript",
                data=transcript_text,
                file_name="video_transcript.txt",
                mime="text/plain",
                key="transcript_btn_final"
            )
    
    # Instructions in sidebar
    with st.sidebar:
        st.header("📖 How it works")
        st.markdown("""
        1. **Enter a YouTube URL**
        2. **Choose summary type:**
           - Single Summary: One comprehensive summary
           - Chapter-wise: Divide content into 3, 5, or 10 chapters
        3. **The app will:**
           - Check for existing subtitles (manual or auto-generated)
           - If found in Hindi, translate to English
           - If no subtitles, transcribe the audio
           - Generate summaries using AI
        4. **Download** the summary or full transcript
        
        **Note:** Audio transcription may take several minutes for longer videos.
        """)
        
        st.header("🚀 Features")
        st.markdown("""
        - ✅ **No API keys required**
        - ✅ **Completely free**
        - ✅ **Supports multiple languages**
        - ✅ **Works with videos without subtitles**
        - ✅ **Single or chapter-wise summaries**
        - ✅ **Adjustable summary length**
        - ✅ **Download results**
        """)
        
        st.header("🛠️ Technology")
        st.markdown("""
        - **Summarization**: BART (Facebook)
        - **Transcription**: Faster Whisper
        - **Translation**: Google Translate
        """)

if __name__ == "__main__":
    main()