import html


def build_camera_placeholder_svg(width, height, image_topic):
    safe_image_topic = html.escape(image_topic)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#eef4f6"/>
  <rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="22" fill="#ffffff" stroke="#d1dde3" stroke-width="3"/>
  <g transform="translate({width / 2},{height / 2 - 28})">
    <rect x="-78" y="-52" width="156" height="104" rx="16" fill="#dfe9ee" stroke="#8fa4ae" stroke-width="3"/>
    <circle cx="0" cy="0" r="22" fill="#8fa4ae"/>
    <path d="M-44 28 L-18 2 L10 30 L28 12 L52 38" fill="none" stroke="#5b7480" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
  <text x="50%" y="{height - 92}" text-anchor="middle" font-family="IBM Plex Sans, Noto Sans KR, sans-serif" font-size="26" font-weight="700" fill="#30414a">Waiting for {safe_image_topic}</text>
  <text x="50%" y="{height - 54}" text-anchor="middle" font-family="IBM Plex Sans, Noto Sans KR, sans-serif" font-size="18" fill="#5b7480">Live frame will appear here</text>
</svg>""".encode('utf-8')


def extract_jpeg_dimensions(frame_bytes):
    if len(frame_bytes) < 4 or frame_bytes[0] != 0xFF or frame_bytes[1] != 0xD8:
        return None, None

    i = 2
    while i + 1 < len(frame_bytes):
        if frame_bytes[i] != 0xFF:
            i += 1
            continue

        while i < len(frame_bytes) and frame_bytes[i] == 0xFF:
            i += 1
        if i >= len(frame_bytes):
            break

        marker = frame_bytes[i]
        i += 1

        if marker in (0xD8, 0xD9):
            continue

        if i + 1 >= len(frame_bytes):
            break

        segment_length = (frame_bytes[i] << 8) | frame_bytes[i + 1]
        if segment_length < 2 or i + segment_length > len(frame_bytes):
            break

        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if segment_length >= 7:
                height = (frame_bytes[i + 3] << 8) | frame_bytes[i + 4]
                width = (frame_bytes[i + 5] << 8) | frame_bytes[i + 6]
                return width, height
            break

        i += segment_length

    return None, None

