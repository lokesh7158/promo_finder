"""Synthetic-only screenplay samples for the hosted demonstration."""

from __future__ import annotations

import io
import zipfile


_EVENT_TYPES = ("ACTION", "CHARACTER", "SOUND FX", "CAMERA", "CHARACTER", "ACTION")
_EVENT_TEXT = (
    "A synthetic figure enters the demonstration room.",
    "The synthetic narrator says, The blue door is open.",
    "A soft synthetic chime sounds.",
    "The camera observes the empty demonstration room.",
    "The synthetic narrator says, We should look inside.",
    "A synthetic light turns on behind the door.",
)


def build_synthetic_screenplay(episode_number: int) -> bytes:
    """Create one harmless parser-compatible synthetic episode."""
    if not 1 <= episode_number <= 50:
        raise ValueError("Synthetic episode number must be between 1 and 50")

    lines = [
        f"EPISODE NUMBER: Episode {episode_number}",
        f"EPISODE LOGLINE: Synthetic demonstration episode {episode_number}.",
        "CONTINUITY NOTES: Synthetic demonstration content only.",
        "",
        "--- SCRIPT START ---",
    ]
    for scene_number, scene_offset in ((1, 0), (2, 60)):
        scene_start = f"{scene_offset // 60:02d}:{scene_offset % 60:02d}.00"
        scene_end_seconds = scene_offset + 60
        scene_end = f"{scene_end_seconds // 60:02d}:{scene_end_seconds % 60:02d}.00"
        lines.extend(
            [
                "",
                f"SCENE {scene_number}",
                "SCENE HEADING: INT. SYNTHETIC DEMONSTRATION ROOM - DAY",
                f"SCENE START: {scene_start}",
                f"SCENE END: {scene_end}",
                "ACTIONS AND DIALOGUE:",
            ]
        )
        for event_index, (event_type, event_text) in enumerate(
            zip(_EVENT_TYPES, _EVENT_TEXT, strict=True)
        ):
            start_seconds = scene_offset + event_index * 10
            end_seconds = start_seconds + 10
            start = f"{start_seconds // 60:02d}:{start_seconds % 60:02d}.00"
            end = f"{end_seconds // 60:02d}:{end_seconds % 60:02d}.00"
            label = "DEMO NARRATOR" if event_type == "CHARACTER" else event_type
            lines.extend(
                [
                    f"**{label}**",
                    f"Start: {start} | End: {end}",
                    f"{event_text} Episode {episode_number}, event {event_index + 1}.",
                    "",
                ]
            )
    lines.extend(["--- SCRIPT END ---", ""])
    return "\n".join(lines).encode("utf-8")


def synthetic_screenplays(count: int = 50) -> list[tuple[str, bytes]]:
    """Return a numbered collection of synthetic screenplay files."""
    if not 1 <= count <= 50:
        raise ValueError("Synthetic sample count must be between 1 and 50")
    return [
        (f"episode-{number:03d}-synthetic-demo.txt", build_synthetic_screenplay(number))
        for number in range(1, count + 1)
    ]


def synthetic_pack_zip(count: int = 50) -> bytes:
    """Build a downloadable zip containing only generated synthetic files."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in synthetic_screenplays(count):
            archive.writestr(filename, content)
    return output.getvalue()
