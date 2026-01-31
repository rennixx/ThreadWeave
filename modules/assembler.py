"""
Video Assembler Module

Combines animated clips and audio into final video.
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class VideoAssembler:
    """
    Assembles final video from animated clips and audio.

    Supports both moviepy and ffmpeg for video assembly.
    """

    def __init__(self, fps: int = 30, resolution: tuple = (1080, 1920)):
        """
        Initialize the video assembler.

        Args:
            fps: Frames per second for output video
            resolution: (width, height) tuple for output resolution
        """
        self.fps = fps
        self.resolution = resolution

        # Video export settings
        self.codec = "libx264"
        self.audio_codec = "aac"
        self.video_bitrate = "8M"
        self.audio_bitrate = "192k"
        self.preset = "medium"

        # Check which method is available
        self._check_available_methods()

        logger.info(f"VideoAssembler initialized: {fps} FPS, resolution={resolution}")

    def _check_available_methods(self) -> None:
        """Check which assembly methods are available."""
        self.moviepy_available = False
        self.ffmpeg_available = False

        # Check for moviepy
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
            self.moviepy_available = True
            logger.info("moviepy is available")
        except ImportError:
            logger.info("moviepy not installed")

        # Check for ffmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                shell=False
            )
            self.ffmpeg_available = result.returncode == 0
            if self.ffmpeg_available:
                logger.info("ffmpeg is available")
        except FileNotFoundError:
            logger.info("ffmpeg not available")

        if not self.moviepy_available and not self.ffmpeg_available:
            raise RuntimeError(
                "No video assembly method available. "
                "Install moviepy or ffmpeg."
            )

    def assemble_video(
        self,
        clip_paths: list[str],
        audio_path: str,
        output_path: str,
        use_fade: bool = True,
        fade_duration: float = 0.5
    ) -> str:
        """
        Assemble final video from clips and audio.

        Args:
            clip_paths: List of video clip file paths
            audio_path: Path to audio file
            output_path: Path to save final video
            use_fade: Whether to add fade transitions
            fade_duration: Duration of fade in/out in seconds

        Returns:
            str: Path to the assembled video

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If assembly fails
        """
        # Validate inputs
        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                raise FileNotFoundError(f"Clip not found: {clip_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        logger.info(f"Assembling {len(clip_paths)} clips into final video...")

        # Create output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Try moviepy first, fallback to ffmpeg
        if self.moviepy_available:
            try:
                return self._assemble_with_moviepy(
                    clip_paths, audio_path, output_path, use_fade, fade_duration
                )
            except Exception as e:
                logger.warning(f"moviepy assembly failed: {e}. Trying ffmpeg...")
                if self.ffmpeg_available:
                    return self._assemble_with_ffmpeg(
                        clip_paths, audio_path, output_path
                    )
                raise

        elif self.ffmpeg_available:
            return self._assemble_with_ffmpeg(
                clip_paths, audio_path, output_path
            )

        else:
            raise RuntimeError("No assembly method available")

    def _assemble_with_moviepy(
        self,
        clip_paths: list[str],
        audio_path: str,
        output_path: str,
        use_fade: bool,
        fade_duration: float
    ) -> str:
        """Assemble video using moviepy."""
        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

        logger.info("Using moviepy for assembly...")

        # Load clips
        clips = []
        for clip_path in clip_paths:
            clip = VideoFileClip(clip_path)
            clips.append(clip)

        # Add fade transitions
        if use_fade:
            clips = self._add_fade_transitions(clips, fade_duration)

        # Concatenate
        final_clip = concatenate_videoclips(clips, method="compose")

        # Add audio
        audio = AudioFileClip(audio_path)
        final_clip = final_clip.set_audio(audio)

        # Export
        final_clip.write_videofile(
            output_path,
            fps=self.fps,
            codec=self.codec,
            audio_codec=self.audio_codec,
            bitrate=self.video_bitrate,
            preset=self.preset,
            logger=None  # Suppress moviepy's progress bar
        )

        # Close clips to free memory
        for clip in clips:
            clip.close()
        final_clip.close()
        audio.close()

        logger.info(f"Video assembled with moviepy: {output_path}")
        return output_path

    def _assemble_with_ffmpeg(
        self,
        clip_paths: list[str],
        audio_path: str,
        output_path: str
    ) -> str:
        """Assemble video using ffmpeg."""
        logger.info("Using ffmpeg for assembly...")

        # Create concat file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            for clip_path in clip_paths:
                # Use absolute paths and escape for ffmpeg
                abs_path = os.path.abspath(clip_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-i', audio_path,
                '-c:v', self.codec,
                '-c:a', self.audio_codec,
                '-b:v', self.video_bitrate,
                '-b:a', self.audio_bitrate,
                '-shortest',
                '-y',  # Overwrite output file
                output_path
            ]

            logger.debug(f"ffmpeg command: {' '.join(cmd)}")

            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

            logger.info(f"Video assembled with ffmpeg: {output_path}")
            return output_path

        finally:
            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)

    def _add_fade_transitions(
        self,
        clips: list,
        fade_duration: float
    ) -> list:
        """
        Add fade in/out transitions to clips.

        Args:
            clips: List of VideoClip objects
            fade_duration: Duration of fade in seconds

        Returns:
            list: Clips with fade transitions applied
        """
        faded_clips = []

        for i, clip in enumerate(clips):
            # First clip: fade in only
            if i == 0:
                faded = clip.fadein(fade_duration)
            # Last clip: fade out only
            elif i == len(clips) - 1:
                faded = clip.fadeout(fade_duration)
            # Middle clips: fade in and out
            else:
                faded = clip.fadein(fade_duration).fadeout(fade_duration)

            faded_clips.append(faded)

        return faded_clips

    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about a video file.

        Args:
            video_path: Path to video file

        Returns:
            dict: Video information (duration, fps, resolution, size, codec)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.moviepy_available:
            return self._get_info_moviepy(video_path)
        elif self.ffmpeg_available:
            return self._get_info_ffmpeg(video_path)
        else:
            return {
                "path": video_path,
                "file_size": os.path.getsize(video_path)
            }

    def _get_info_moviepy(self, video_path: str) -> dict:
        """Get video info using moviepy."""
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)

        info = {
            "path": video_path,
            "duration": clip.duration,
            "fps": clip.fps,
            "resolution": (clip.w, clip.h),
            "file_size": os.path.getsize(video_path)
        }

        clip.close()
        return info

    def _get_info_ffmpeg(self, video_path: str) -> dict:
        """Get video info using ffprobe (part of ffmpeg)."""
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate',
                    '-show_entries', 'format=duration',
                    '-of', 'json',
                    video_path
                ],
                capture_output=True,
                text=True,
                check=False
            )

            import json

            if result.returncode == 0:
                data = json.loads(result.stdout)
                video_stream = data.get('streams', [{}])[0]
                format_info = data.get('format', {})

                # Parse fps (e.g., "30/1" -> 30.0)
                fps_str = video_stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str)

                return {
                    "path": video_path,
                    "duration": float(format_info.get('duration', 0)),
                    "fps": fps,
                    "resolution": (
                        int(video_stream.get('width', 0)),
                        int(video_stream.get('height', 0))
                    ),
                    "file_size": os.path.getsize(video_path)
                }

        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")

        # Fallback
        return {
            "path": video_path,
            "file_size": os.path.getsize(video_path)
        }

    def create_horizontal_version(
        self,
        vertical_path: str,
        output_path: str
    ) -> str:
        """
        Create a horizontal (16:9) version from vertical video.

        Pads the sides with black bars.

        Args:
            vertical_path: Path to vertical (9:16) video
            output_path: Path to save horizontal video

        Returns:
            str: Path to horizontal video
        """
        if not self.ffmpeg_available:
            raise RuntimeError("ffmpeg required for horizontal conversion")

        # Target: 1920x1080 (16:9)
        # Input: 1080x1920 (9:16)
        # Strategy: Scale to fit height, then pad sides

        cmd = [
            'ffmpeg',
            '-i', vertical_path,
            '-vf', 'scale=1080:-1,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
            '-c:a', 'copy',
            '-y',
            output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        logger.info(f"Horizontal version created: {output_path}")
        return output_path
