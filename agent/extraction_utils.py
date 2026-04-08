import fitz
from pathlib import Path
from urllib.parse import urlparse, urlsplit, unquote
import base64
from PIL import Image
from io import BytesIO
import io
import os
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks import get_openai_callback

import cv2
import numpy as np
import pdfplumber
from markitdown import MarkItDown
from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions, ContentSettings
from datetime import datetime, timedelta
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd
import os.path
import re
import aiohttp
import aiofiles
import time

import gc
import asyncio

from models import init_llm, GPT54_args, GPT54m_args

from typing import Dict, List, Union, Literal, Optional, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict


class Note(BaseModel):
    progress: Literal["completed", "continue"] = Field(
        ..., description="Extraction status"
    )
    remarks: str = Field(
        ..., description="Detailed extraction progress and resume instructions"
    )

    model_config = ConfigDict(extra="forbid")

class ExtractionOutput(BaseModel):
    extraction_progress: Optional[Dict[str, Union[str, List[Dict[str, str]]]]] = Field(
        None,
        description="Newly extracted data from this run. Values can be a string or a list of row objects."
    )
    note: Optional[Note] = Field(
        None, description="Extraction progress metadata"
    )

    model_config = ConfigDict(extra="forbid")
    

cv2.setUseOptimized(True)
EnhanceMode = Literal["none", "fast", "heavy"]


async def load_file_to_bytes_async(source: str, local_path: str = None) -> tuple[BytesIO, str]:
    """Asynchronously download or read a file from a URL or local path into a BytesIO buffer.

    Async counterpart of ``load_file_to_bytes``. Uses aiohttp for HTTP fetches
    and aiofiles for local disk reads so the event loop is never blocked.

    Args:
        source: A URL (http/https) or local file path to the source file.
        local_path: If provided, the fetched file bytes are also written here
            asynchronously.

    Returns:
        A tuple of (BytesIO buffer containing the file data, file type string
        without the leading dot, e.g. "pdf", "xlsx").

    Raises:
        TypeError: If the file extension is not in the accepted list.
        ValueError: If fetching from a URL fails.
        IOError: If reading from or saving to a local path fails.
    """
    accepted_extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.gif', '.bmp', '.tiff', '.xls', '.xlsx']
    
    # Parse the URL to extract the path
    parsed_url = urlparse(source)
    if parsed_url.scheme in ('http', 'https'):
        # If it's a URL, extract the path and then the file extension
        path = urlsplit(source).path
        file_extension = Path(unquote(path)).suffix.lower()
    else:
        # If it's a local path, directly get the file extension
        file_extension = Path(source).suffix.lower()

    if file_extension not in accepted_extensions:
        raise TypeError("Unsupported file type. Only .pdf, .jpg, .jpeg, .png, .gif, .bmp, .tiff, .xls, and .xlsx are accepted.")
    
    file_type = file_extension.replace(".", "")

    try:
        if parsed_url.scheme in ('http', 'https'):
            # Fetch the file from the URL asynchronously using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    response.raise_for_status()
                    content = await response.read()
                    file_data = BytesIO(content)
        else:
            # Read the file from the local file path asynchronously using aiofiles
            async with aiofiles.open(source, 'rb') as file:
                content = await file.read()
                file_data = BytesIO(content)
    except aiohttp.ClientError as e:
        raise ValueError(f"Error fetching the file from the URL: {str(e)}")
    except Exception as e:
        raise IOError(f"Error reading the file from the local path: {str(e)}")

    # Save the file to local path if provided, asynchronously
    if local_path:
        try:
            async with aiofiles.open(local_path, 'wb') as file:
                await file.write(file_data.getbuffer())
        except Exception as e:
            raise IOError(f"Error saving the file to the local path: {str(e)}")

    return file_data, file_type


def is_base64_image(string):
    """Check whether a string is a valid base64-encoded image.

    Attempts to decode the string and verify it against known image file
    signatures (JPEG, PNG). Falls back to PIL verification for other formats.

    Args:
        string: A candidate base64-encoded string to test.

    Returns:
        True if the string decodes to a valid image, False otherwise.
    """
    try:
        # Attempt to decode the base64 string
        image_data = base64.b64decode(string, validate=True)
        
        # Quick check for common image signatures (headers)
        if image_data.startswith(b'\xFF\xD8\xFF'):  # JPEG signature
            return True
        elif image_data.startswith(b'\x89PNG\r\n\x1A\n'):  # PNG signature
            return True
        # Add other formats if needed (e.g., GIF, BMP)
        
        # Fallback to full verification if header check is inconclusive
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify that it is an image
        return True
    except (base64.binascii.Error, IOError):
        return False


def convert_buffer_to_base64_list(buffer, file_type):
    """Convert a file buffer (PDF or image) into a list of base64-encoded page images.

    For PDFs, each page is rendered via PyMuPDF and encoded separately.
    For single images, the buffer is encoded as one element.

    Args:
        buffer: A BytesIO or file-like object containing the file data.
        file_type: The file extension string (e.g. "pdf", "png", "jpg").

    Returns:
        A list of base64-encoded strings, one per page/image.
    """
    base64_list = []
    
    if file_type == 'pdf':
        pdf_document = fitz.open(stream=buffer, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_buffer = BytesIO(pix.tobytes())
            base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')
            base64_list.append(base64_image)
            img_buffer.close()  # Release temporary buffer       
        pdf_document.close()  # Release PDF document memory
    else:
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        base64_list.append(base64_image)
    
    return base64_list


def image_to_base64(image: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """
    Converts a NumPy image array to a base64-encoded string.

    Parameters:
        image (numpy.ndarray): The image to encode.
        fmt (str): Image format, e.g. ".jpg" or ".png".
        quality (int): JPEG quality (1-100). Ignored for PNG.

    Returns:
        base64_string (str): The base64-encoded image string.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    _, buffer = cv2.imencode(fmt, image, encode_params)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string


async def detect_orientation(images_to_detect: list[str]) -> dict:
    """Detect the rotation needed and blank-page status for each scanned page image.

    Sends each base64-encoded page image to a vision LLM, which returns the
    clockwise rotation (0/90/180/270 degrees) required to make the page upright,
    along with whether the page is essentially blank.

    Args:
        images_to_detect: List of base64-encoded image strings, one per page.

    Returns:
        A list of dicts, one per image, each containing:
            - "rotation" (int): Clockwise degrees to apply (0, 90, 180, or 270).
            - "reason" (str): Brief justification for the rotation choice.
            - "is_blank" (bool): True if the page has no meaningful content.
    """
    base_prompt = """
You are given a single scanned document page image.

Return a JSON object that specifies:
1) the CLOCKWISE rotation (in degrees) that must be APPLIED to the image to make it upright
2) whether the page is essentially blank

IMPORTANT: The "rotation" value is the CORRECTION TO APPLY (not the current orientation).

------------------------------------------------------------
1) ROTATION TO APPLY (CLOCKWISE CORRECTION)
------------------------------------------------------------

Choose "rotation" from: 0, 90, 180, 270.

Interpretation:
- rotation = the number of degrees to rotate the page CLOCKWISE so the result is upright and readable.

Upright means:
- For English / Simplified Chinese business documents: most text lines are horizontal and read left-to-right.
- Text is not upside down.
- Ignore small logos/stamps that may be skewed; follow the dominant body text/table orientation.
- For vertical East Asian typesetting (vertical columns ordered right-to-left), pick the orientation where those columns look natural.

Direction constraint:
- Always output a CLOCKWISE rotation value only.
- If the correction needed visually feels "rotate left / counter-clockwise", convert it to the equivalent clockwise value:
  - 90° CCW correction  == 270° CW rotation
  - 180° CCW correction == 180° CW rotation
  - 270° CCW correction == 90° CW rotation

Reason requirement:
- In "reason", justify the chosen CLOCKWISE correction using visible cues (header text, table rows, logos, etc.).
- Avoid phrasing that sounds like applying counter-clockwise rotation. If mentioning left/CCW, explicitly state it is equivalent to the chosen clockwise number.

------------------------------------------------------------
2) BLANK PAGE DECISION
------------------------------------------------------------

Decide whether the page is essentially blank.

"is_blank" is true ONLY if there is NO meaningful content such as:
- printed/handwritten text, numbers
- tables, lines
- stamps, signatures
- barcodes / QR codes
- logos / letterheads
- even small codes/page numbers make it NOT blank

If ANY meaningful mark/text/code exists, then "is_blank" must be false.

------------------------------------------------------------
3) OUTPUT FORMAT (STRICT)
------------------------------------------------------------

Return ONLY valid JSON with EXACTLY this structure:

{{
  "rotation": 0,
  "reason": "brief justification of the clockwise correction",
  "is_blank": false
}}

Rules:
- Output MUST be valid JSON.
- Use double quotes for all keys and string values.
- "rotation" MUST be a number (0/90/180/270), not a string.
- "is_blank" MUST be a boolean, not a string.
- Do not output any text outside the JSON.
"""

    ai_role = "You are an excellent document analyzer."
        
    # Construct the prompt and input dict in the same loop
    combined_prompt = [{"type": "text", "text": base_prompt}]
    
    img_token = "image_data"
    image_prompt = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{{{img_token}}}"
    }
    # Append the image prompt to the combined prompt
    combined_prompt.append(image_prompt)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ai_role),
            ("user", combined_prompt),
        ]
    )

    selected_llm = GPT54m_args.copy()
    selected_llm['reasoning_effort'] = 'high'
    llm = init_llm(selected_llm)
    chain = prompt | llm | JsonOutputParser()

    chain_params = []
    for image in images_to_detect:
        params = {}
        params['image_data'] = image
        chain_params.append(params)

    start_time = time.time()
    response_list = await chain.abatch(chain_params, config={"max_concurrency": 50})
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"detect_orientation @ elapsed_time: {elapsed_time:.2f} seconds")

    return response_list


def rotate_to_upright(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate a BGR image clockwise by the specified angle to make it upright.

    Args:
        image_bgr: The BGR image as a NumPy ndarray.
        angle: Clockwise rotation in degrees. Must be one of 0, 90, 180, or 270.

    Returns:
        The rotated BGR image.

    Raises:
        ValueError: If angle is not in {0, 90, 180, 270}.
    """
    if angle == 0:
        return image_bgr
    elif angle == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def make_blank_placeholder(size: int = 10) -> np.ndarray:
    """Create a small white placeholder image for blank pages.

    Args:
        size: The width and height in pixels of the square placeholder.

    Returns:
        A white (255, 255, 255) BGR NumPy ndarray of shape (size, size, 3).
    """
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def expand_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    scale: float = 1.05,   # 5% larger in each dimension
) -> tuple[float, float, float, float]:
    """Expand a normalized bounding box around its center by a scale factor.

    Useful for adding a safety margin around detected content regions.
    Output values are clamped to [0.0, 1.0]. If expansion produces an
    invalid box, the original coordinates are returned unchanged.

    Args:
        x_min: Left edge (normalized, 0.0-1.0).
        y_min: Top edge (normalized, 0.0-1.0).
        x_max: Right edge (normalized, 0.0-1.0).
        y_max: Bottom edge (normalized, 0.0-1.0).
        scale: Multiplicative factor for width and height.
            1.0 means no change; 1.05 means ~5% larger.

    Returns:
        A tuple (x_min, y_min, x_max, y_max) of the expanded box.
    """
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = (x_max - x_min) * scale
    h = (y_max - y_min) * scale

    x_min_new = max(0.0, cx - w / 2.0)
    y_min_new = max(0.0, cy - h / 2.0)
    x_max_new = min(1.0, cx + w / 2.0)
    y_max_new = min(1.0, cy + h / 2.0)

    # Ensure still valid
    if x_max_new <= x_min_new or y_max_new <= y_min_new:
        return x_min, y_min, x_max, y_max

    return x_min_new, y_min_new, x_max_new, y_max_new


def opencv_cropping(
    image: np.ndarray,
    kernel_size: int = 10,
    offset: bool = False,
    offset_measure: int = 30,
    blank_ratio_threshold: float = 1e-5,   # <=0.001% non-white pixels ⇒ blank
    min_contour_area: int = 50,            # ignore very tiny blobs
) -> np.ndarray:
    """Detect the main non-white content region in an image and crop to it.

    Uses morphological operations and contour detection on a contrast-enhanced
    grayscale version to find the largest non-white area. The crop is applied
    to the original (unenhanced) image to preserve colors.

    Args:
        image: BGR input image as a NumPy ndarray.
        kernel_size: Size of the morphological closing kernel used to merge
            nearby content regions.
        offset: Whether to add an extra margin around the detected region.
        offset_measure: Pixel margin to add on each side when offset is True.
        blank_ratio_threshold: If the fraction of non-white pixels is below
            this value, the page is considered blank.
        min_contour_area: Minimum contour area in pixels; smaller contours
            are treated as noise and ignored.

    Returns:
        The cropped BGR image, or a 10x10 white placeholder if the page is
        essentially blank or has no detectable content.
    """
    # ----------------------------------------------------------------------
    # STEP 0: sanity check
    # ----------------------------------------------------------------------
    if image is None or image.size == 0:
        placeholder = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return placeholder

    # ----------------------------------------------------------------------
    # STEP 1: LIGHT GLOBAL CONTRAST FOR CROPPING ONLY
    # ----------------------------------------------------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)  # used only for mask/white detection
    lab_eq = cv2.merge((l_eq, a, b))
    contrast_for_mask = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    lab = l = a = b = l_eq = lab_eq = None  # free references

    gray = cv2.cvtColor(contrast_for_mask, cv2.COLOR_BGR2GRAY)
    contrast_for_mask = None

    # ----------------------------------------------------------------------
    # STEP 2: FIND NON-WHITE AREA FOR CROPPING
    # ----------------------------------------------------------------------
    white_min = 240
    white_max = 255

    # Binary mask where non-white areas are 255
    _, binary_mask = cv2.threshold(gray, white_min, white_max, cv2.THRESH_BINARY_INV)
    gray = None

    # Robust blank-page detection
    nonwhite_pixels = cv2.countNonZero(binary_mask)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    nonwhite_ratio = nonwhite_pixels / total_pixels if total_pixels > 0 else 0.0

    if nonwhite_ratio < blank_ratio_threshold:
        # Essentially blank → small white placeholder
        placeholder = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return placeholder

    # Morphological closing to expand/merge content regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    expanded_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = None

    # Find contours in the expanded mask
    contours, _ = cv2.findContours(
        expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    expanded_mask = None

    if not contours:
        placeholder = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return placeholder

    # Filter out tiny contours (noise)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
    if not large_contours:
        placeholder = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return placeholder

    # Largest contour = main content region
    largest_contour = max(large_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if offset:
        x -= offset_measure
        y -= offset_measure
        w += 2 * offset_measure
        h += 2 * offset_measure

    # Clamp to image bounds
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Crop from ORIGINAL image to preserve original color/background
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image


def increase_contrast(
    image: np.ndarray,
    enhance_mode: EnhanceMode = "fast",    # "none" | "fast" | "heavy"
    contrast_skip_threshold: Optional[float] = 35.0,
) -> np.ndarray:
    """Enhance the contrast of a BGR image without any cropping.

    Operates on the L (lightness) channel in LAB colour space. If the image
    already has sufficient contrast (measured by grayscale standard deviation),
    enhancement is skipped to avoid over-processing.

    Args:
        image: BGR input image as a NumPy ndarray.
        enhance_mode: Enhancement strategy to apply:
            - "none": Return the original image unchanged.
            - "fast": Apply CLAHE on the L channel (good speed/quality trade-off).
            - "heavy": Apply NLM denoising, unsharp mask, then CLAHE (slow).
        contrast_skip_threshold: If not None and enhance_mode is not "none",
            the grayscale standard deviation of the image is computed. If it
            exceeds this threshold the image is returned as-is (already
            high-contrast).

    Returns:
        The enhanced BGR image, or a 10x10 white placeholder if the input
        is None or empty.

    Raises:
        ValueError: If enhance_mode is not one of "none", "fast", or "heavy".
    """
    # ----------------------------------------------------------------------
    # STEP 0: sanity check
    # ----------------------------------------------------------------------
    if image is None or image.size == 0:
        placeholder = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return placeholder

    # ----------------------------------------------------------------------
    # STEP 1: OPTIONAL CONTRAST-BASED EARLY EXIT ON ORIGINAL IMAGE
    # ----------------------------------------------------------------------
    if contrast_skip_threshold is not None and enhance_mode != "none":
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(gray_full.std())
        # If already high contrast, skip heavy work
        if contrast > contrast_skip_threshold:
            return image

    # ----------------------------------------------------------------------
    # STEP 2: OPTIONAL ENHANCEMENT
    # ----------------------------------------------------------------------
    if enhance_mode == "none":
        return image

    # Convert to LAB for L-channel manipulation
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)

    if enhance_mode == "fast":
        # FAST: CLAHE only on L (no denoising, no unsharp)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_enh = clahe.apply(L)

    elif enhance_mode == "heavy":
        # HEAVY: denoise + unsharp mask + CLAHE (slow)
        L_denoised = cv2.fastNlMeansDenoising(
            L, None, h=8, templateWindowSize=7, searchWindowSize=21
        )

        blur_L = cv2.GaussianBlur(L_denoised, (0, 0), sigmaX=1.0)
        L_sharp = cv2.addWeighted(L_denoised, 1.6, blur_L, -0.6, 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_enh = clahe.apply(L_sharp)

        L_denoised = None
        blur_L = None
        L_sharp = None

    else:
        raise ValueError(f"Unknown enhance_mode={enhance_mode!r}")

    lab_enh = cv2.merge((L_enh, A, B))
    A = B = L_enh = None

    enhanced_bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    lab_enh = None

    return enhanced_bgr


async def process_file_async(
    file_data: io.BytesIO,
    file_type: str,
    output_dir: str = None,
    kernel_size: int = 500,
    offset: bool = True,
    offset_measure: int = 20,
    max_output_size_mb: int = 5,
    page_numbers: Optional[List[int]] = None,  # 1-based PDF page numbers
) -> List[str]:
    """Process a PDF or image file into a list of cleaned, orientation-corrected base64 page images.

    Pipeline per page: render -> OpenCV crop whitespace -> LLM orientation
    detection -> rotate to upright -> contrast enhancement -> size enforcement.
    Blank pages (detected by the orientation LLM) are replaced with a small
    white placeholder.

    Args:
        file_data: BytesIO buffer containing the source file.
        file_type: File extension string (e.g. "pdf", "png", "jpg").
        output_dir: If provided, processed images are also saved here as PNGs.
        kernel_size: Morphological kernel size for OpenCV whitespace cropping.
        offset: Whether to add a pixel margin around the detected content.
        offset_measure: Pixel margin size when offset is True.
        max_output_size_mb: Maximum allowed size (MB) for each output image;
            images exceeding this are iteratively downscaled.
        page_numbers: Optional 1-based page numbers to process from a PDF.
            If None or empty, all pages are processed.

    Returns:
        A list of base64-encoded PNG strings, one per processed page/image.
        Returns an empty list if the file type is unsupported or processing fails.
    """

    def is_file_size_large_pil(image: Image.Image, max_size_mb: int = 20) -> bool:
        """Check if a PIL image exceeds the given size limit when saved as PNG."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        size_mb = buffer.tell() / (1024 * 1024)
        return size_mb > max_size_mb

    def resize_if_needed_pil(
        image: Image.Image, max_width: int = 2000, max_height: int = 2000
    ) -> Image.Image:
        """Downscale a PIL image to fit within max dimensions, preserving aspect ratio."""
        if image.width > max_width or image.height > max_height:
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            print(f"Image resized to fit within {max_width}x{max_height}.")
        return image

    def ensure_output_size_cv(
        image_bgr: np.ndarray,
        max_size_mb: int = 5,
        min_dim: int = 300,
        max_dim: int = 1999,
        refine_factor: float = 0.85,
    ) -> np.ndarray:
        
        # Target 95% of the max limit to leave room for the rest of the JSON request payload
        safe_max_mb = max_size_mb * 0.95 

        # --- Enforce max pixel dimension ---    
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = max(min_dim, int(w * scale))
            new_h = max(min_dim, int(h * scale))
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Pixel cap: resized {w}x{h} -> {new_w}x{new_h}")
            
        # --- Enforce file size limit ---
        def get_b64_size_mb(img: np.ndarray) -> float:
            """Encodes to JPEG and calculates exact Base64 payload size in MB"""
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # 1 character in a base64 string = 1 byte of payload
            b64_len = len(base64.b64encode(buffer))
            return b64_len / (1024 * 1024)

        size_mb = get_b64_size_mb(image_bgr)
        if size_mb <= safe_max_mb:
            return image_bgr

        print(f"Base64 JPEG size ({size_mb:.2f} MB) exceeds limit, resizing...")

        resized_img = image_bgr.copy()
        
        while size_mb > safe_max_mb and resized_img.shape[1] > min_dim and resized_img.shape[0] > min_dim:
            # Estimate how much to shrink based on the area (quadratic scaling)
            scale = (safe_max_mb / max(size_mb, 1e-6)) ** 0.5
            scale *= 0.95 # Extra buffer
            
            # Force it to shrink by at least the refine_factor to prevent infinite loops
            scale = min(scale, refine_factor)
            
            new_w = max(min_dim, int(resized_img.shape[1] * scale))
            new_h = max(min_dim, int(resized_img.shape[0] * scale))
            
            if (new_w, new_h) == (resized_img.shape[1], resized_img.shape[0]):
                break
                
            # INTER_AREA is the highest quality algorithm for downscaling in OpenCV
            resized_img = cv2.resize(resized_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            size_mb = get_b64_size_mb(resized_img)
            print(f"Resized to {new_w}x{new_h}, Base64 JPEG size now {size_mb:.2f} MB")

        if size_mb > max_size_mb:
            print(f"Warning: Could not reduce JPEG below {max_size_mb} MB "
                f"without shrinking below {min_dim}x{min_dim}. Final size: {size_mb:.2f} MB")

        return resized_img

    def normalize_selected_pages(total_pages: int, pages: Optional[List[int]]) -> List[int]:
        """Return a deduplicated, in-range list of 1-based page numbers to process."""
        if not pages:
            return list(range(1, total_pages + 1))

        seen = set()
        result: List[int] = []
        for p in pages:
            if not isinstance(p, int):
                continue
            if p in seen:
                continue
            seen.add(p)
            if 1 <= p <= total_pages:
                result.append(p)
        return result

    async def run_orientation_async(inputs: List[str]) -> List[dict]:
        """Run detect_orientation as an async task with exception handling."""
        task = asyncio.create_task(detect_orientation(inputs))
        res = await asyncio.gather(task, return_exceptions=True)
        out0 = res[0]
        if isinstance(out0, Exception):
            print(f"detect_orientation failed: {out0}")
            return []
        return out0 or []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    extension = f".{(file_type or '').strip().lower().lstrip('.')}"
    file_content = file_data.getvalue()
    base64_images: List[str] = []

    if extension == ".pdf":
        images = None
        try:
            info = await asyncio.to_thread(pdfinfo_from_bytes, file_content)
            total_pages = int(info.get("Pages", 0))
            if total_pages <= 0:
                return []

            selected_page_nums = normalize_selected_pages(total_pages, page_numbers)
            if not selected_page_nums:
                print("No valid page_numbers provided; returning empty result.")
                return []

            page_bgr_list: List[np.ndarray] = []
            b64_for_ai: List[str] = []
            original_page_nums: List[int] = []

            for page_num in selected_page_nums:
                pil_page = None
                one_page_images = None
                try:
                    one_page_images = await asyncio.to_thread(
                        convert_from_bytes,
                        file_content,
                        dpi=200,
                        first_page=page_num,
                        last_page=page_num,
                    )
                    if not one_page_images:
                        continue

                    pil_page = one_page_images[0]
                    if is_file_size_large_pil(pil_page):
                        print(f"Page {page_num} is too large, resizing input...")
                        pil_page = resize_if_needed_pil(pil_page)

                    page_bgr = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
                    page_bgr_list.append(page_bgr)
                    b64_for_ai.append(image_to_base64(page_bgr))
                    original_page_nums.append(page_num)

                finally:
                    if pil_page is not None and hasattr(pil_page, "close"):
                        try:
                            pil_page.close()
                        except Exception as e:
                            print(f"Error closing PIL image for page {page_num}: {e}")
                    pil_page = None
                    one_page_images = None

            file_content = None

            num_pages = len(page_bgr_list)
            if num_pages == 0:
                return []

            orientation_list = await run_orientation_async(b64_for_ai)

            if len(orientation_list) != num_pages:
                print(
                    "Warning: detect_orientation result length does not "
                    "match number of selected pages. Some pages will use defaults."
                )

            for sel_i, page_bgr in enumerate(page_bgr_list):
                page_num = original_page_nums[sel_i]
                processed = None
                try:
                    if sel_i < len(orientation_list):
                        orient = orientation_list[sel_i] or {}
                        is_blank = bool(orient.get("is_blank", False))
                        current_rot = int(
                            orient.get("rotation", orient.get("current_rotation", 0)) or 0
                        )
                    else:
                        print(
                            f"Page {page_num}: missing orientation result, "
                            "assuming not blank & rotation=0"
                        )
                        is_blank = False
                        current_rot = 0

                    if is_blank:
                        processed = make_blank_placeholder()
                    else:
                        cropped = opencv_cropping(
                            image=page_bgr,
                            kernel_size=kernel_size,
                            offset=offset,
                            offset_measure=offset_measure,
                        )
                        processed = rotate_to_upright(cropped, int(current_rot))
                        processed = increase_contrast(
                            processed,
                            enhance_mode="fast",
                            contrast_skip_threshold=35.0,
                        )

                    processed = ensure_output_size_cv(processed, max_output_size_mb)

                    if output_dir:
                        output_path = os.path.join(output_dir, f"page_{page_num}_processed.jpeg")
                        cv2.imwrite(output_path, processed)
                        print(f"Processed image saved to {output_path}")

                    base64_images.append(image_to_base64(processed))

                finally:
                    page_bgr = None
                    processed = None

        except Exception as e:
            print(f"Error processing PDF pages: {e}")

        finally:
            images = None
            gc.collect()

        return base64_images

    if extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        pil_image = None
        try:
            file_data.seek(0)
            pil_image = Image.open(file_data)

            print("Processing single image...")

            if is_file_size_large_pil(pil_image):
                print("Input image is too large, resizing...")
                pil_image = resize_if_needed_pil(pil_image)

            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            b64_for_ai = [image_to_base64(image_bgr)]

            orientation_list = await run_orientation_async(b64_for_ai)

            if orientation_list:
                orient = orientation_list[0] or {}
                print(f"Orientation result: {orient}")
                is_blank = bool(orient.get("is_blank", False))
                current_rot = int(
                    orient.get("rotation", orient.get("current_rotation", 0)) or 0
                )
            else:
                print("No orientation result; assuming not blank & rotation=0")
                is_blank = False
                current_rot = 0

            if is_blank:
                processed = make_blank_placeholder()
            else:
                cropped = opencv_cropping(
                    image=image_bgr,
                    kernel_size=kernel_size,
                    offset=offset,
                    offset_measure=offset_measure,
                )
                processed = rotate_to_upright(cropped, int(current_rot))
                processed = increase_contrast(
                    processed,
                    enhance_mode="fast",
                    contrast_skip_threshold=35.0,
                )

            processed = ensure_output_size_cv(processed, max_output_size_mb)

            if output_dir:
                output_path = os.path.join(output_dir, "image_processed.jpeg")
                cv2.imwrite(output_path, processed)
                print(f"Processed image saved to {output_path}")

            base64_images.append(image_to_base64(processed))

        except Exception as e:
            print(f"Error processing image: {e}")

        finally:
            if pil_image is not None and hasattr(pil_image, "close"):
                try:
                    pil_image.close()
                except Exception as e:
                    print(f"Error closing PIL image: {e}")
            pil_image = None
            gc.collect()

        return base64_images

    print(f"Unsupported file format: {extension}")
    return []


def restructure_to_markdown(data):
    """Convert a list of table data (from pdfplumber) into a Markdown string.

    Each table is a list of rows, where each row is a list of cell values.
    Empty tables are skipped. The output is a series of pipe-delimited
    Markdown table blocks separated by blank lines.

    Args:
        data: A list of tables, where each table is a list of rows
            (list of lists of cell values).

    Returns:
        A Markdown-formatted string containing all non-empty tables.
    """
    def remove_newlines(table):
        """Replace newlines in all cells with spaces."""
        return [[str(item).replace("\n", " ") if item else "" for item in row] for row in table]

    def format_table(table):
        """Format a single table (list of rows) as a Markdown table string."""
        table_str = []
        headers = table[0]
        rows = table[1:]

        # Calculate the maximum length of each column
        max_lengths = [len(str(item)) for item in headers]
        for row in rows:
            for i, item in enumerate(row):
                max_lengths[i] = max(max_lengths[i], len(str(item)))

        # Format the headers
        headers_str = "| " + " | ".join([str(headers[i]).ljust(max_lengths[i]) for i in range(len(headers))]) + " |"
        separator_str = "| " + " | ".join(["-" * max_lengths[i] for i in range(len(headers))]) + " |"
        
        table_str.append(headers_str)
        table_str.append(separator_str)
        
        # Format the rows
        for row in rows:
            row_str = "| " + " | ".join([str(row[i]).ljust(max_lengths[i]) for i in range(len(row))]) + " |"
            table_str.append(row_str)
        
        return "\n".join(table_str)

    markdown_str = ""

    # Process each table in the data
    for table in data:
        table = remove_newlines(table)
        if not any(any(row) for row in table):  # Skip empty tables
            continue

        # Format the header table
        for row in table:
            if any(row):
                markdown_str += "| " + " | ".join([str(item) if item else "" for item in row]) + " |\n"
        
        markdown_str += "\n"

    return markdown_str


def extract_content_with_ocr(
    file_data: io.BytesIO,
    file_type: str,
    page_numbers: list[int] | None
) -> str:
    """Extract text and tables from a PDF using pdfplumber OCR as a supplementary reference.

    For each selected page, extracts tabular content as Markdown tables and
    non-table text (character-by-character with spatial layout heuristics).
    The combined result is used as a supporting OCR reference alongside
    vision-based extraction.

    Note: Curly braces in the output are escaped ({{ / }}) so the result can
    be safely embedded in LangChain prompt templates.

    Args:
        file_data: BytesIO buffer containing the PDF file.
        file_type: Must be "pdf"; returns None for any other type.
        page_numbers: Optional list of 1-based page numbers to process.
            If None or empty, all pages are processed.

    Returns:
        A string with page-separated OCR content (Markdown tables + text),
        or None if the file is not a PDF or processing fails entirely.
    """
    if file_type != "pdf":
        return None

    try:
        with pdfplumber.open(file_data) as pdf:
            total_pages = len(pdf.pages)

            # If None/empty -> all pages (1-based page numbers)
            if not page_numbers:
                selected_pages = list(range(1, total_pages + 1))
            else:
                # Keep order, remove duplicates while preserving order
                seen = set()
                selected_pages = []
                for p in page_numbers:
                    if p not in seen:
                        seen.add(p)
                        selected_pages.append(p)

            # Optional: filter out-of-range pages (change to raise if desired)
            selected_pages = [p for p in selected_pages if 1 <= p <= total_pages]

            ocr_result_list: list[tuple[int, str]] = []
            text_limit = 10000

            for page_num in selected_pages:
                page = pdf.pages[page_num - 1]  # pdfplumber is 0-based indexing here

                try:
                    tables = page.extract_tables()
                    table_bboxes = [t.bbox for t in page.find_tables()]

                    # Extract text outside tables
                    text = ""
                    sorted_chars = sorted(page.chars, key=lambda c: (c["top"], c["x0"]))

                    previous_top = None
                    previous_x0 = None

                    for char in sorted_chars:
                        in_table = any(
                            bbox[0] <= char["x0"] <= bbox[2] and bbox[1] <= char["top"] <= bbox[3]
                            for bbox in table_bboxes
                        )
                        if in_table:
                            continue

                        if previous_top is not None:
                            margin = abs(char["top"] - previous_top)
                            line_breaks_num = margin // 5
                            total_line_breaks = min(int(line_breaks_num), 2)
                            text += "\n" * total_line_breaks

                        if previous_x0 is not None:
                            horizontal_margin = abs(char["x0"] - previous_x0)
                            if horizontal_margin > 10:
                                text += " "

                        text += char["text"]
                        previous_top = char["top"]
                        previous_x0 = char["x0"]

                    markdown_table = restructure_to_markdown(tables)  # uses existing helper

                    parts = []
                    if len(markdown_table) > 0 and len(text) == 0:
                        parts.append(markdown_table)
                    elif 0 < len(text) < text_limit and len(markdown_table) == 0:
                        parts.append(text)
                    elif 0 < len(text) < text_limit and len(markdown_table) > 0:
                        parts.append(markdown_table + "\n" + text)

                    content = "\n".join(parts).strip()
                    if content:
                        ocr_result_list.append((page_num, content))

                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    ocr_result_list.append((page_num, f"Error processing page {page_num}: {str(e)}"))

        batch_contents = [f"**page number: {page_num}**\n{content}" for page_num, content in ocr_result_list]
        ocr_result = "\n\n".join(batch_contents)
        ocr_result = ocr_result.replace("{", "{{").replace("}", "}}")
        return ocr_result

    except Exception as e:
        print(f"Error processing PDF file: {str(e)}")
        return None


async def extract_content_with_vlm(
    encoded_images: list[str],
    selected_fields: list[str],
    page_numbers: list[int] = None,
    ocr_result: str = None,
    extraction_progress: dict = None,
    long_input: bool = False,
    selected_llm_args: dict = GPT54_args
) -> dict:
    """Extract structured data from document page images using a vision language model.

    Constructs a detailed prompt with the field schema, OCR reference text,
    extraction progress (for iterative/chunked runs), and base64 page images,
    then sends it to the configured Azure OpenAI vision model. Supports
    iterative extraction for large tables (50-row chunks) and long documents
    processed in page batches.

    Args:
        encoded_images: List of base64-encoded page image strings.
        selected_fields: List of field definitions describing what to extract,
            including field names, data types, and optional remarks.
        page_numbers: 1-based page numbers corresponding to the images,
            included in the prompt for the LLM's context.
        ocr_result: Optional pdfplumber OCR text to supplement the vision
            extraction for better character accuracy.
        extraction_progress: Previous extraction state dict (with keys
            "extraction_progress" and "note") for continuation runs.
            None for a fresh extraction.
        long_input: If True, adds chunk-awareness instructions so the LLM
            scopes its progress/completion flags to the visible chunk only.
        selected_llm_args: LLM configuration dict (defaults to GPT-5.4).

    Returns:
        A dict with "extraction_progress" (newly extracted data) and "note"
        (progress metadata including per-field completion status and remarks).
    """
    if extraction_progress:
        note = extraction_progress.get('note', '')
        extraction_states = extraction_progress.get('extraction_progress') or {}
        previous_progress = {
            k: v[-5:] if isinstance(v, list) else v 
            for k, v in extraction_states.items()
        }
    else:
        note = "This is a new extraction"
        previous_progress = "No progress yet"
        
    progress_description = (
        f"Progress status:\n{note}\n\n"
        f"Previous Results (showing the latest 5 rows only if table):\n{previous_progress}"
    )
    
    if ocr_result is not None:
        ocr_prompt = f"""
You will have a preliminary OCR result as a supporting reference.

**OCR Result for reference:**

{ocr_result}

Use this OCR text mainly for:
- Character accuracy of IDs, item/style codes, names, and numbers.
- Helping to read text that may be small or slightly unclear in the image.

Do NOT rely on the OCR layout (line breaks, spacing, or column alignment) as the primary indicator of table structure.
"""
    else:
        ocr_prompt = ""
        
    if long_input:
        long_input_prompt = """
**Further Instructions for Chunked / Long Document Extraction**

The content and images provided in this call are only a subset (a chunk) of a longer document.
Other pages may exist before and after the pages you see here.

You must still follow all previous instructions, especially:
- The Expected Output Structure (single-instance vs multiple-instance).
- Rules for `extraction_progress`, `note.progress`, and `note.remarks`.
- Large-table handling (50-row limit, resume points, etc.).

This section only clarifies how to apply those rules in a chunked context.

-----------------------------------------
1. Scope of `note.progress` in a chunk
-----------------------------------------

Always interpret `completed` / `incomplete` / `missing` with respect to the content that is
actually visible in this chunk:

- `completed`:
  - You have extracted all relevant content for that field/table that is visible in this chunk.
  - Do NOT mark a field/table as `incomplete` just because the full document might continue on
    pages you do not see.

- `incomplete`:
  - Relevant content for that field/table is visible in this chunk but has not been fully
    extracted in this run.
  - Typical reasons:
    * You reached the 50-row limit for a large table, so additional visible rows remain
      unprocessed.
    * The progress description or your current run stops at a clear resume point while further
      visible rows/records for that field/table are still present.

- `missing`:
  - No relevant content for that field/table is visible anywhere in this chunk.

-----------------------------------------
2. Chunk-aware use of `note.remarks`
-----------------------------------------

In `note.remarks`, explain how the extraction you performed relates to this specific chunk:

- For fields/tables marked `completed`:
  - State that they are complete **with respect to this chunk**.
  - If the visible part looks like the beginning, middle, or end of a larger section/table
    (e.g., starts mid-period or stops at a page boundary), mention that and, if possible,
    indicate the first and last visible identifiers (such as dates, IDs, PO numbers).

- For fields/tables marked `incomplete`:
  - Explain why they are incomplete in this chunk (e.g., 50-row limit reached, visible rows
    beyond the resume point not yet processed).
  - When possible, name:
    * The first and last visible row/record you processed in this run, and
    * What remains visible but unprocessed (using IDs, dates, or other anchors).

- For fields/tables marked `missing`:
  - Explicitly state that no content relevant to that field/table appears in the visible pages
    of this chunk.

Whenever helpful, also:
- Refer to the page numbers in this chunk.
- Indicate whether what you see is likely the beginning, middle, or end of a larger logical unit
  (table, section, record sequence).

These chunk-aware remarks and progress flags will be used by another assistant to merge results
from multiple chunks into a globally complete extraction.
"""
    else:
        long_input_prompt = ""

    base_template = """
You are a highly skilled expert in document analysis, text extraction, and data structuring across multiple languages. Your expertise lies in interpreting and organizing complex, unstructured content from diverse document formats into structured, actionable data for analysis. 
You excel at recognizing varied layouts, identifying key information, and ensuring extracted data is precise, contextually accurate, and faithful to the original language and intent.

You are provided with a list of fields to extract, each following this fixed structure:  
- **`field name:`** → The exact key name to be used in the JSON output.  
  When locating the corresponding value in the content, you do not need to match this text exactly — instead, identify the most relevant and contextually correct value for this field.  
- **`data type:`** → The required output format for the extracted value (e.g., Text, Date, Table).  
  If the data type is `Table`, return the extracted data in JSON as an **array of dictionaries** — this does not necessarily mean the source contains a table.  
- **Optional `remarks:`** → Additional guidance, such as:  
  * A more detailed description of the value to extract.  
  * If the field name refers to a table in the content, the column names for that table (to be used as keys in each row’s dictionary in the JSON output).  
  * Formatting rules (e.g., date format, numeric formatting, unit conversions).

Additionally, you will have access to images of the document for processing.

---

**Extraction Progress and Continuity Instructions:**
- You will be provided with an **Extraction Progress description** that explains:  
    - Which fields/tables have been completed.  
    - Which fields/tables are incomplete.  
    - For each incomplete table/field:  
        * Provide the exact **last extracted row** using a unique combination of column values from the source.
        * If the next row to process is visible in the provided content, explicitly name it by listing its unique combination of column values as the **resume starting point**.  
        * If no further rows for the current table are visible in the provided content, mark the table as **completed** in the progress notes and do not include a resume point.  
        * Only include a resume point if there is clear evidence of continuation (e.g., partial row, header for next section visible).  
        * State why the extraction stopped (e.g., chunk size limit, end of provided content, table ended). 

- You must **attempt extraction for all fields/tables listed**, even if some values are missing in the provided content.

---

**Generic End-of-Table Detection for Image-based Tables:**  
When processing tables from image-based documents (e.g., scanned bank statements, invoices, receipts, reports):

- **Header Identification:**  
  1. Identify the first row that clearly defines the table’s column structure (e.g., labels such as “Date”, “Description”, “Amount”, “Balance”, “Item”, “Qty”, “Price”, etc.).  
  2. Record the number and order of columns from this header.

- **Row Continuation Rules:**  
  - Continue extracting rows that match the expected column count and logical structure.  
  - Allow for multi-line cells (e.g., long descriptions) as part of the same row.  
  - Preserve correct mapping of values to their respective columns.

- **End-of-Table Conditions:**  
  Stop extracting when **any** of the following occur:
  1. **Column mismatch** — A row has significantly fewer or more columns than the header after cleaning text.  
  2. **Layout break** — A large gap, section divider, or clear change in format/headers indicates a new table or section.  
  3. **Consecutive non-data rows** — Two or more rows contain only whitespace, decorative symbols, or irrelevant text.  
  4. **Summary or footer cues** — Row contains keywords that indicate the end of the table, such as:  
     - “Total”, “Grand Total”  
     - “Closing Balance” / “Opening Balance”  
     - “End of Statement”  
     - “Subtotal”  
     - “Page Total” / “Carried Forward”  
     - “Summary” / “Remarks”  
  5. **Non-tabular content** — Footer text, disclaimers, contact info, page numbers, or marketing messages.

- **Post-End Handling:**  
  - Once an end condition is met, mark the table as **completed**.  
  - Ignore unrelated content after the table unless explicitly requested in the selected fields.  
  - Do not include totals or summaries unless they are part of the requested extraction.

---

**Expected Output Structure:**  
- `extraction_progress` → A dictionary containing only the newly extracted data from this run. Keys = field names or table names, values = JSON-compatible types.  
- `note` → A dictionary with two keys:   
    - `remarks` → Detailed progress notes in free-text describing:
        * Which fields/tables are completed.
        * Which remain incomplete and the exact resume point.
        * Any constraints or pending work.
    - `progress` → A dictionary where:
        * Key = field/table name
        * Value = one of `"completed"`, `"incomplete"`, `"missing"`
        * `"completed"` → Field/table fully extracted and no further data visible.
        * `"incomplete"` → Partial extraction done, visible resume point exists.
        * `"missing"` → Field/table value not found in provided content and no resume point exists; empty value returned.

---

**Rules for Setting `note.progress`:**
- For each field/table processed in this run:
    * If fully extracted → `"completed"`.
    * If partially extracted and resume point exists → `"incomplete"`.
    * If missing and no resume point → `"missing"`.
- Do not skip any fields in this run — mark each one with a status.

---

**General Instructions:**

1. **Analysis:**  
    - Review the entire document, including all sections, appendices, and tabular data.  
    - Ensure accurate interpretation and no misrepresentation.  

2. **Data Identification & Extraction:**  
    - Extract the most relevant and complete value for each field.  
    - Cross-check for consistency.  
    - For any table output, return an array of dictionaries with keys as specified in `remarks` or `field name`.  
    - Include all rows with at least one non-empty cell.  

3. **Language & Sensitivity:**  
    - Preserve original language unless translation is requested.  
    - Follow local formatting conventions for dates, numbers, and currencies if specified.  

4. **Output Standards:**  
    - Remove thousand separators, currency symbols, and percentage signs unless required.  
    - Output must be valid JSON.  
    - Do not invent or fabricate values.  
    - For missing values:  
        * Return `""` for single-value fields.  
        * Return `[]` for tables.  
    - For tables:  
        * The JSON key must directly contain the array of row objects (no redundant nesting).  

---

**Special Handling for Large Tables:**

1. Chunk Size Rule
   - Regardless of whether the table is shown in full or partially in the provided content, extract at most 50 rows in this run.
   - Always begin from the first unprocessed row (based on the resume point from previous progress notes).
   - Stop after 50 rows, even if more rows remain visible in the input.
   - Remaining rows will be extracted in subsequent runs.

2. Row Counting
   - Count rows based on actual records output in this run (ignore header and separator lines).

3. Stopping at Limit
   - Stop immediately at 50 rows, even if you are in the middle of a logical group or section.

4. Filling the Chunk
   - When enough rows are available, fill the output to as close to 50 rows as possible.

5. Small Group Handling
   - Do not stop extraction after a small group unless it is the final visible group in the table.

6. Fewer than 50 Rows Remaining
   - If fewer than 50 rows remain from the resume point to the end of the table, extract all of them and mark the table as "completed".

7. Resume Point Rule
   - Only set a resume point if the next unprocessed row is explicitly visible in the provided content.

8. Avoid Duplicate Extraction
   - Never re-extract rows already processed in prior runs unless explicitly instructed.
   
---

**Objective:**  
Deliver accurate, structured data extraction in strict adherence to these rules.  
Return **only** valid JSON in the specified structure, with no extra commentary.

---

{long_input_prompt}

**Fields to Extract and Specific Instructions:**  
{selected_fields}

**Extraction Progress:**  
{progress_description}

{ocr_prompt}

Page Number of the pages in the document provided:
{page_numbers}
"""

    prompt_parts = [{"type": "text", "text": base_template}]
    params = {
        "selected_fields": selected_fields,
        "progress_description": progress_description,
        "ocr_prompt": ocr_prompt,
        "page_numbers": page_numbers,
        "long_input_prompt": long_input_prompt
    }

    # Only add successfully processed images as image_url parts
    for i, image_data in enumerate(encoded_images, start=1):
        var_name = f"image_data{i}"
        prompt_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{{{var_name}}}"})
        params[var_name] = image_data

    prompt_template = HumanMessagePromptTemplate.from_template(template=prompt_parts)
    prompt = ChatPromptTemplate.from_messages([prompt_template])
    
    llm = init_llm(selected_llm_args)
    chain = prompt | llm | JsonOutputParser()
    
    response = await chain.ainvoke(params)

    return response


def convert_text_from_excel(file_data):
    """Convert an Excel file to Markdown text using the MarkItDown library.

    Args:
        file_data: A file path string or file-like object accepted by
            MarkItDown.convert().

    Returns:
        A Markdown string representation of the Excel content.
    """
    md = MarkItDown()
    content = md.convert(source=file_data)

    return content.markdown


def clean_markdown_tables(md_text, placeholder="<EMPTY>"):
    """Clean Markdown tables by replacing empty/NaN/unnamed cells and removing all-empty columns.

    Parses the Markdown text to identify table blocks (header + separator + rows),
    replaces blank, "nan", and "Unnamed*" cell values with a placeholder, then
    drops any column where all data rows contain only the placeholder.
    Non-table lines pass through unchanged.

    Args:
        md_text: A Markdown-formatted string potentially containing tables.
        placeholder: The string to substitute for empty/invalid cell values.

    Returns:
        The cleaned Markdown string with sanitized tables.
    """
    lines = md_text.splitlines()
    cleaned_lines = []
    inside_table = False
    current_table = []  # store lines for the current table

    def process_table(table_lines):
        """Clean a single Markdown table block: replace empty cells and drop all-empty columns."""
        # Convert to list of cell lists
        rows = [ [cell.strip() for cell in line.split("|")] for line in table_lines ]

        # Replace unwanted values with placeholder
        for r in range(len(rows)):
            for c in range(len(rows[r])):
                cell = rows[r][c]
                if (cell == "" or cell.lower() == "nan" or cell.lower().startswith("unnamed")):
                    rows[r][c] = placeholder

        # Identify columns where all data rows (excluding header & separator) are placeholder
        col_empty_flags = [True] * len(rows[0])
        for r in range(2, len(rows)):  # start from row 2: skip header & separator
            for c in range(len(rows[r])):
                if rows[r][c] != placeholder and rows[r][c] != "":
                    col_empty_flags[c] = False

        # Remove columns that are all-empty
        cleaned_rows = []
        for r in range(len(rows)):
            cleaned_row = [rows[r][c] for c in range(len(rows[r])) if not col_empty_flags[c]]
            cleaned_line = "|".join(cleaned_row)
            if not cleaned_line.startswith("|"):
                cleaned_line = "|" + cleaned_line
            if not cleaned_line.endswith("|"):
                cleaned_line = cleaned_line + "|"
            cleaned_rows.append(cleaned_line)

        return cleaned_rows

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of a table
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r'^\s*\|?\s*-+', lines[i + 1]):
            inside_table = True
            current_table = [line]  # start collecting table lines
            continue

        # Detect separator row inside table
        if inside_table and re.match(r'^\s*\|?\s*-+', stripped):
            current_table.append(line)
            continue

        # Process table rows
        if inside_table and stripped.startswith("|"):
            current_table.append(line)
            continue

        # End of table block when we hit non-table text
        if inside_table and not stripped.startswith("|"):
            inside_table = False
            # Process the collected table and append cleaned result
            cleaned_lines.extend(process_table(current_table))
            current_table = []

        # Non-table lines remain untouched
        cleaned_lines.append(line)

    # If file ends with a table, process it
    if inside_table and current_table:
        cleaned_lines.extend(process_table(current_table))

    return "\n".join(cleaned_lines)


async def extract_content_from_excel(
    excel_content: list[dict],
    selected_fields: list[str],
    extraction_progress: dict = None,
    selected_llm_args: dict = GPT54_args
) -> dict:
    """Extract structured data from Excel spreadsheet content using an LLM.

    Similar to ``extract_content_with_vlm`` but operates on text-based Excel
    content (converted to Markdown) rather than images. Supports iterative
    extraction with a 50-row-per-run chunk limit for large tables, and tracks
    progress across multiple invocations.

    Args:
        excel_content: List of dicts representing the Excel content, typically
            sheet-level Markdown text with page/sheet identifiers.
        selected_fields: List of field definitions describing what to extract.
        extraction_progress: Previous extraction state dict for continuation
            runs. None for a fresh extraction.
        selected_llm_args: LLM configuration dict (defaults to GPT-5.4).

    Returns:
        A dict with "extraction_progress" (newly extracted data) and "note"
        (progress metadata including per-field completion status and remarks).
    """
    template = """
You are a highly skilled expert in document analysis, text extraction, and data structuring across multiple languages. Your expertise lies in interpreting and organizing complex, unstructured content from diverse document formats into structured, actionable data for analysis.
You excel at recognizing varied layouts, identifying key information, and ensuring extracted data is precise, contextually accurate, and faithful to the original language and intent.

You are provided with a list of fields to extract, each following this fixed structure:  
Fields to Extract:
[
    `field name 1`: `remarks 1 `,
    `field name 2`: `remarks 2`,
    ...
]
where:
- **`field name:`** (before `:`) → The exact key name to be used in the JSON output.  
  When locating the corresponding value in the content, you do not need to match this text exactly — instead, identify the most relevant and contextually correct value for this field. 
- **Key normalization (important):**
  If a field name appears wrapped in quotes or code formatting (e.g., `'MILL_NAME'`, `"MILL_NAME"`, or `` `MILL_NAME` ``),
  treat those quote/backtick characters as formatting only.
  Use the unwrapped text as the JSON key (e.g., use `MILL_NAME`, not `'MILL_NAME'`).
  Always strip only the outermost wrapping quotes/backticks; do not alter internal characters. 
- **Optional `remarks:` (after `:`)** → Additional guidance, such as:  
  * A more detailed description of the value to extract.  
  * If the field name refers to a table in the content, the column names for that table (to be used as keys in each row’s dictionary in the JSON output).  
  * Formatting rules (e.g., date format, numeric formatting, unit conversions).
(If you see user require the values should be returned in a list of dictionaries meaning the extracted data in JSON as an **array of dictionaries**)

Additionally, you will have access to content extracted from an Excel spreadsheet for processing.

BUSINESS_RULES (highest priority):
In addition to accurately extracting values, you must apply the provided BUSINESS_RULES as mandatory constraints that govern:
- which source evidence to prefer when multiple candidates exist,
- how to normalize/format extracted values (dates, amounts, units, casing),
- how to resolve ambiguities and conflicts across the document,
- and when to leave fields empty due to insufficient evidence.

BUSINESS_RULES always override general extraction heuristics and any conflicting interpretation of field names/remarks.
If a BUSINESS_RULE cannot be satisfied because the required evidence is missing or ambiguous, do not guess; return an empty value for the affected field(s) ("", or [] for tables).
When extracting any field, treat BUSINESS_RULES as part of the field’s extraction instructions (as if they were included in that field’s remarks).

---

**Extraction Progress and Continuity Instructions:**
- You will be provided with an **Extraction Progress description** that explains:  
    - Which fields/tables have been completed.  
    - Which fields/tables are incomplete.  
    - For each incomplete table/field:  
        * Provide the exact **last extracted row** using a unique combination of column values from the source.
        * If the next row to process is visible in the provided content, explicitly name it by listing its unique combination of column values as the **resume starting point**.  
        * If no further rows for the current table are visible in the provided content, mark the table as **completed** in the progress notes and do not include a resume point.  
        * Only include a resume point if there is clear evidence of continuation (e.g., partial row, header for next section visible).  
        * State why the extraction stopped (e.g., chunk size limit, end of provided content, table ended). 

- You must **attempt extraction for all fields/tables listed**, even if some values are missing in the provided content.

- **Generic End-of-Table Detection:**  
  When processing a Markdown table, identify its header row (above the `|---|---|...|` separator).  
  The table ends when:  
    - The next line does not have the same number of columns as the header, OR  
    - A new separator row appears, OR  
    - Consecutive empty or `<EMPTY>` rows appear with no meaningful data.  
  Once the end is reached, mark the table as completed and ignore any unrelated summary/total rows after it.

---

**Expected Output Structure:**  
- `extraction_progress` → A dictionary containing only the newly extracted data from this run. Keys = field names or table names, values = JSON-compatible types.  
- `note` → A dictionary with two keys:
    - `remarks` → Detailed progress notes in free-text describing:
        * Which fields/tables are completed.
        * Which remain incomplete and the exact resume point.
        * Any constraints or pending work.
    - `progress` → A dictionary where:
        * Key = field/table name
        * Value = one of `"completed"`, `"incomplete"`, `"missing"`
        * `"completed"` → Field/table fully extracted and no further data visible.
        * `"incomplete"` → Partial extraction done, visible resume point exists.
        * `"missing"` → Field/table value not found in provided content and no resume point exists; empty value returned.

---

**Rules for Setting `note.progress`:**
- For each field/table processed in this run:
    * If fully extracted → `"completed"`.
    * If partially extracted and resume point exists → `"incomplete"`.
    * If missing and no resume point → `"missing"`.
- Do not skip any fields in this run — mark each one with a status.

---

**General Instructions:**

1. **Analysis:**  
    - Review the entire document, including all sections, appendices, and tabular data.  
    - Ensure accurate interpretation and no misrepresentation.  

2. **Data Identification & Extraction:**  
    - Extract the most relevant and complete value for each field.  
    - Cross-check for consistency.  
    - For any table output, return an array of dictionaries with keys as specified in `remarks` or `field name`.  
    - Include all rows with at least one non-empty cell.  

3. **Language & Sensitivity:**  
    - Preserve original language unless translation is requested.  
    - Follow local formatting conventions for dates, numbers, and currencies if specified.  

4. **Output Standards:**  
    - Remove thousand separators, currency symbols, and percentage signs unless required.  
    - Output must be valid JSON.  
    - Do not invent or fabricate values.  
    - For missing values:  
        * Return `""` for single-value fields.  
        * Return `[]` for tables.  
    - For tables:  
        * The JSON key must directly contain the array of row objects (no redundant nesting).  

---

**Special Handling for Large Tables:**

1. Chunk Size Rule
   - Regardless of whether the table is shown in full or partially in the provided content, extract at most 50 rows in this run.
   - Always begin from the first unprocessed row (based on the resume point from previous progress notes).
   - Stop after 50 rows, even if more rows remain visible in the input.
   - Remaining rows will be extracted in subsequent runs.

2. Row Counting
   - Count rows based on actual records output in this run (ignore header and separator lines).

3. Stopping at Limit
   - Stop immediately at 50 rows, even if you are in the middle of a logical group or section.

4. Filling the Chunk
   - When enough rows are available, fill the output to as close to 50 rows as possible.

5. Small Group Handling
   - Do not stop extraction after a small group unless it is the final visible group in the table.

6. Fewer than 50 Rows Remaining
   - If fewer than 50 rows remain from the resume point to the end of the table, extract all of them and mark the table as "completed".

7. Resume Point Rule
   - Only set a resume point if the next unprocessed row is explicitly visible in the provided content.

8. Avoid Duplicate Extraction
   - Never re-extract rows already processed in prior runs unless explicitly instructed.

---

**Objective:**  
Deliver accurate, structured data extraction in strict adherence to these rules.  
Return **only** valid JSON in the specified structure, with no extra commentary.

---

**Extraction Progress:**  
{progress_description}

**Page Number and Content of Excel Spreadsheet:**  
{excel_content}

**Fields to Extract and Specific Instructions:**  
{selected_fields}
"""


    llm = init_llm(selected_llm_args)
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=['selected_fields', 'excel_content', 'progress_description']
    )

    chain = prompt | llm | JsonOutputParser()
    
    if extraction_progress:
        note = extraction_progress.get('note', '')
        extraction_states = extraction_progress.get('extraction_progress') or {}
        previous_progress = {
            k: v[-5:] if isinstance(v, list) else v 
            for k, v in extraction_states.items()
        }
    else:
        note = "This is a new extraction"
        previous_progress = "No progress yet"
        
    progress_description = (
        f"Progress status:\n{note}\n\n"
        f"Previous Results (showing the latest 5 rows only if table):\n{previous_progress}"
    )
        
    response = await chain.ainvoke(
        {
            'selected_fields': selected_fields, 
            'excel_content': excel_content, 
            'progress_description': progress_description
            }
        )
    
    return response


def merge_dicts(old_dict, new_dict):
    """Recursively merge new_dict into old_dict, mutating old_dict in place.

    Merge rules per key:
    - Both values are lists: extend old list with new items.
    - Both values are dicts: merge recursively.
    - Otherwise (type mismatch or scalar): overwrite with new value.
    - Key only in new_dict: add it to old_dict.

    Args:
        old_dict: The base dictionary to merge into (modified in place).
        new_dict: The dictionary whose entries are merged into old_dict.

    Returns:
        The mutated old_dict with merged contents.
    """
    for key, new_value in new_dict.items():
        if key in old_dict:
            old_value = old_dict[key]
            # If both values are lists, extend the old list with the new
            if isinstance(old_value, list) and isinstance(new_value, list):
                old_dict[key].extend(new_value)
            # If both values are dictionaries, recursively merge them
            elif isinstance(old_value, dict) and isinstance(new_value, dict):
                merge_dicts(old_value, new_value)
            # Otherwise, overwrite with new value
            else:
                old_dict[key] = new_value
        else:
            # If key doesn't exist in old_dict, add it
            old_dict[key] = new_value
    return old_dict


async def consolidate_extractions(
    batch_extractions: List[str],
    selected_fields: List[str],
) -> tuple[dict[str, Any] | list[dict[str, Any]], dict[str, Any]]:
    """Consolidate extraction results from multiple page-batch chunks into a single coherent result.

    Used for long documents that are processed in batches (e.g. >50 pages).
    Sends all per-batch extraction results plus the field schema to a reasoning
    LLM, which merges them into one final result while handling section-aware
    deduplication, multi-instance documents, and table continuation.

    Args:
        batch_extractions: List of serialized extraction result dicts (one per
            page batch), each containing "extraction_progress" and "note".
        selected_fields: The field definitions used during extraction, so the
            LLM understands the target schema.

    Returns:
        A tuple of (consolidated_result, token_usage) where:
        - consolidated_result is a dict (single-instance) or list of dicts
          (multi-instance) containing the merged extraction data.
        - token_usage is a dict with "total_tokens", "prompt_tokens", and
          "completion_tokens".
    """
    template = """
You are an expert at consolidating structured extraction results from long, chunked documents.

Context:
- You are given multiple **extraction result sets**.
- Each set comes from a different chunk (set of pages) of the **same underlying document**.
- Each set normally has this structure (or similar):

  {{
    "extraction_progress": {{ ... extracted fields/tables for that chunk ... }},
    "note": {{
      "progress": {{ ... per-field statuses, including "multiple_extractions" ... }},
      "remarks": "chunk-aware explanations of what was extracted, where it came from (section/pages/anchors), and what may be missing"
    }}
  }}

- The `Fields to Extract and Specific Instructions` describe the schema for **one logical document/record** (e.g., one purchase order, one invoice, one lease record). When multiple instances of that record type are present in a document, the extractor applies this schema once per instance.

Your objective is to consolidate all these per-chunk extraction results into one **final, coherent result** that matches the requested schema.

CRITICAL PROBLEM TO HANDLE:
- The same field/table key can appear in multiple distinct sections of the document with different meanings/instances (e.g., multiple "Rent Structure" tables).
- The user/instructions may require selecting the value/table from a SPECIFIED SECTION (e.g., “only use the table from 附件五：租金及管理费支付时间表”), not merging across sections.
- `extraction_results` contains provenance clues in `note.remarks` (and sometimes in extracted values) such as section titles, appendix names, page ranges, table titles, continuation statements, and resume points.

You MUST be section-aware and provenance-aware when consolidating.

-----------------------------------------
1. Understand the fields and schema
-----------------------------------------

- Carefully read **Fields to Extract and Specific Instructions**:
  - Understand what each field represents, including any formatting or table column requirements.
  - Identify which fields are scalar values (e.g., `Tenant`, `Area`) and which are tables/arrays (e.g., `Rent Structure`).
  - Identify any explicit constraints like:
    - “Extract ONLY from section/appendix X”
    - “Use the rent structure table from <named section>”
    - “Ignore tables outside <named section>”
  These constraints act as a hard filter.

-----------------------------------------
2. Understand the per-chunk extraction results & provenance
-----------------------------------------

For each set in **Initial extraction results per set**:
- Inspect:
  - `extraction_progress` → extracted values.
  - `note.progress` and `note.remarks` → statuses and provenance.
- Extract provenance signals from `note.remarks` for each field/table, such as:
  - Section/appendix name (e.g., “附件五：租金及管理费支付时间表”)
  - Page numbers or page ranges (e.g., “p.52-54”)
  - Table title/anchor phrases
  - Continuation cues (e.g., “Continued extraction…”, “resume point…”, “table appears to continue…”)
  - Any per-field mention like “Rent Structure: …(section/page/anchor…)”

Treat provenance as authoritative. Do not assume two values belong together just because they share the same key name.

-----------------------------------------
3. Decide single-instance vs multiple-instance consolidation
-----------------------------------------

Use the following rules to decide the final shape of the consolidated result:

- **Multiple-instance case (array output)**
  - If any set has an array of record objects, or it is clear that multiple instances exist (e.g., multiple leases/units with distinct identifiers),
    then the final consolidated result must be an array of record objects.
  - Group and merge objects that belong to the same logical instance (e.g., same tenant/unit/contract identifier), combining fields/tables.

- **Single-instance case (object output)**
  - If there is only one logical instance across all sets and no multi-instance output, the final consolidated result must be a single object.

-----------------------------------------
4. SECTION-AWARE OVERRIDE (HIGHEST PRIORITY)
-----------------------------------------

When the same field/table appears in multiple sections (evidenced by `note.remarks` showing different section/appendix/table titles or different page regions):
- DO NOT automatically merge them.

Instead, apply this strict precedence:

A) If the **Specific Instructions** specify a required section/appendix/table anchor for a field/table:
   1) Select candidates ONLY from that required section/anchor (as evidenced by provenance in `note.remarks`).
   2) Discard candidates from other sections entirely (do not merge, do not deduplicate across them).
   3) If the required section candidate spans multiple chunks, merge ONLY those chunks that:
      - explicitly reference the same required section/appendix name AND
      - indicate continuation/resume of the same table instance (e.g., “continued extraction”, “resume point”, sequential page ranges).

B) If the instructions do NOT specify a section, but multiple distinct section candidates exist:
   1) Prefer the candidate whose provenance indicates it matches the field’s semantic context best:
      - e.g., “Rent Structure” should prefer a section explicitly about rent schedule/payment timetable over a generic summary section.
   2) If still ambiguous, prefer the candidate with:
      - clearer section/table title in remarks,
      - more complete extraction (more rows/columns for tables; more filled related fields for scalars),
      - explicit continuation chain across chunks (resume points, continued extraction).
   3) DO NOT merge candidates from different sections unless provenance clearly shows they are the SAME table instance continued (same section title/anchor).

C) If no provenance exists to distinguish sections:
   - Fall back to normal consolidation rules (completeness-based), but be conservative about merging tables.

-----------------------------------------
5. Consolidating scalar fields (non-table, per-record fields)
-----------------------------------------

For scalar fields (e.g., `Tenant`, `Leasing Type`, `Area`, dates, fees):
- Collect all candidate values across sets.
- If the field is constrained to a required section by instructions:
  - Only consider candidates whose provenance matches that section (see Section-Aware Override).
- Prefer values from sets where:
  - `note.progress[field_name]` is "completed", and
  - the set contains more related fields for the same record (more complete context).
- If multiple sets give different values:
  - Choose the value that best matches the required section/anchor (if applicable).
  - Otherwise choose the one with highest contextual completeness and consistency.

For multi-instance consolidation:
- Do the above per record instance (group by identifier).

-----------------------------------------
6. Consolidating tables (multi-page / multi-chunk)
-----------------------------------------

For table fields (arrays of row objects, e.g., `Rent Structure`):
1) Gather all candidate tables for that field from all sets.
2) Build “table instances” by grouping candidates using provenance extracted from `note.remarks`, such as:
   - same section/appendix name,
   - same table title/anchor,
   - compatible/continuous page ranges,
   - “continued extraction/resume point” relationship.

Section constraint handling:
- If instructions require a specific section/table source for the table:
  - select the matching table instance group ONLY.
  - ignore all other groups.

Merge strategy within ONE table instance:
- Combine rows in logical document order (set 1 then set 2 etc., unless provenance indicates different ordering).
- Avoid duplicates:
  - Consider rows duplicates if all key columns match (e.g., for rent schedule: payment date + period + amount).
- Continuation handling:
  - Use resume points described in `note.remarks` (e.g., last extracted row descriptors) to prevent overlap duplication.

NEVER merge tables across different section/table-instance groups unless provenance explicitly proves they are the same continued table.

-----------------------------------------
7. Using `note.remarks` and `note.progress` to fill gaps
-----------------------------------------

- Use `note.progress` to understand completed/incomplete/missing per chunk.
- Use `note.remarks` to:
  - identify the section/appendix/table title,
  - detect table continuation/resume points,
  - detect that a field is not visible in a chunk.

Goal:
- Maximize completeness while respecting section constraints.
- Do not overwrite correct-section data with wrong-section data.
- Do not “union” tables from different sections when a section constraint exists.

-----------------------------------------
8. Output cleaning (exclude notes/remarks)
-----------------------------------------

- The final result must not include any `note`, `progress`, or `remarks` keys.
- Only include consolidated data according to the schema.

-----------------------------------------
9. Final JSON shape
-----------------------------------------

The final JSON must be structured as:

{{
  "result": <consolidated_result>
}}

Where:
- <consolidated_result> is an array of objects for multi-instance documents, or an object for single-instance documents.

Do not include any text or explanation outside this JSON.

-----------------------------------------
Inputs
-----------------------------------------

Fields to Extract and Specific Instructions:
{selected_fields}

Initial extraction results per set:
{extraction_results}
"""

    selected_llm_args = GPT54_args.copy()
    selected_llm_args['reasoning_effort'] = 'medium'
    
    llm = init_llm(selected_llm_args)
    prompt = PromptTemplate(template=template, input_variables=['extraction_results', 'selected_fields'])
    chain = prompt | llm | JsonOutputParser()

    params = {
        'extraction_results': batch_extractions, 
        'selected_fields': selected_fields
    }

    start_time = time.time()
    with get_openai_callback() as cb:
        response = await chain.ainvoke(params)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    token_usage = {
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
    }
    
    print(f"consolidate_extractions @ elapsed_time: {elapsed_time:.2f} seconds")

    return response, token_usage


# Azure Blob Service Client
async def upload_blob_and_get_url(
    container_name: str, 
    blob_name: str, 
    data: Union[bytes, str], 
    blob_service_client: BlobServiceClient,
    expiry: int = 7,
    content_type: str = None
) -> str:
    """Upload a blob to Azure Blob Storage asynchronously and return a SAS-token URL.

    Creates the target container if it does not already exist. Generates a
    read-only SAS token valid for the specified number of days and appends it
    to the blob URL.

    Args:
        container_name: The name of the Azure Blob Storage container.
        blob_name: The name (key) to assign to the blob in storage.
        data: The content to upload (bytes or string).
        blob_service_client: An async Azure BlobServiceClient instance.
        expiry: SAS token validity period in days. Defaults to 7.
        content_type: Optional MIME type for the blob's Content-Type header.

    Returns:
        The full blob URL with an appended SAS token for read access.

    Raises:
        RuntimeError: If the upload or SAS token generation fails.
    """
    try:
        # Get the container client asynchronously
        container_client = blob_service_client.get_container_client(container_name)

        # Check if container exists, and create if it doesn't
        container_exists = await container_client.exists()
        if not container_exists:
            await container_client.create_container()

        # Upload the blob with content settings asynchronously
        blob_client = container_client.get_blob_client(blob_name)
        
        if content_type is None:
            await blob_client.upload_blob(data, overwrite=True)
        else:
            await blob_client.upload_blob(
                data, 
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )

        # Generate a SAS token that's valid for the specified expiry period
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now() + timedelta(days=expiry)
        )

        # Construct the full URL
        blob_url = f"{blob_client.url}?{sas_token}"

        return blob_url

    except Exception as e:
        # Handle exceptions and provide a meaningful error message
        raise RuntimeError(f"An error occurred while uploading the blob: {str(e)}")
    
    
async def separate_documents_by_image(encoded_images, selected_llm_args: dict = GPT54m_args):
    """Use a vision LLM to partition a multi-document file into separate logical documents.

    Sends all page images to the LLM, which analyses layout, headers, and
    content to group consecutive pages into distinct documents. Useful when
    a single PDF contains multiple documents (e.g. several invoices scanned
    together).

    Args:
        encoded_images: List of base64-encoded page image strings.
        selected_llm_args: LLM configuration dict (defaults to GPT-5.4-mini).

    Returns:
        A dict with a "document_index" key containing a list of dicts, each
        with "page_indices" (list of 0-based page indices) and "reason"
        (explanation for the grouping).
    """
    human_prompt = f"""You are tasked with analyzing a single file containing multiple documents, which has been converted into a series of base64 encoded images, each representing a page. 
Your goal is to identify the document type for each page and separate the pages into distinct documents based on their types.

There are two possible scenarios:
1. **Same Type, Many Documents**: All pages belong to multiple documents of the same type.
2. **Many Types, Many Documents**: Pages belong to multiple documents of varying types.

Steps to Follow:
**Carefully Analyze Each Page**:
   - Examine the text, layout, and any distinctive visual features (e.g., logos, headers, formats) to accurately determine the document type.
   - Note that text may be in languages other than English; make an effort to understand and categorize it based on context and visual cues.

**Separate Documents by Page Index**:
   - The total number of pages in the file is: {len(encoded_images)}
   - Page indices start at 0.
   - Group consecutive pages that belong to the same document.
   - Use the page index to indicate the range of pages for each identified document.

Key Points:
- Ensure that the separation of documents is logical, considering transitions between document (e.g., a new document might start with a cover page or a header).

Output Requirements:
Return the result in JSON format as follows:
dict(
  "document_index": [
    dict(
      "page_indices": [list of page indices belonging to this document],
      "reason": "A concise explanation for why these pages are grouped as this document type."
    ),
    ...
  ]
)
"""

    ai_role = "Your role is to validate the type of the document by thoroughly examining the content and characteristics of the document."
        
    # Construct the prompt and input dict in the same loop
    combined_prompt = [{"type": "text", "text": human_prompt}]
    input_dict = {}
    for i, image_data in enumerate(encoded_images, start=1):
        img_token = f"image_data{i}"
        image_prompt = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{{{img_token}}}"
        }
        combined_prompt.append(image_prompt)
        input_dict[img_token] = image_data
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ai_role),
            ("user", combined_prompt),
        ]
    )
    
    llm = init_llm(selected_llm_args)
    chain = prompt | llm | JsonOutputParser()
    response = await chain.ainvoke(input_dict)

    return response


def split_pdf(pdf_bytes_io: BytesIO, page_groups: list[list[int]]) -> list[BytesIO]:
    """Split a PDF into multiple smaller PDFs based on page index groupings.

    Out-of-range page indices are silently skipped with a warning printed
    to stdout.

    Args:
        pdf_bytes_io: A BytesIO object containing the input PDF.
        page_groups: A list of lists, where each inner list contains 0-based
            page indices that should be combined into one output PDF.

    Returns:
        A list of BytesIO objects, each containing a split PDF corresponding
        to one page group.
    """
    # Read the PDF from BytesIO
    pdf_reader = PdfReader(pdf_bytes_io)
    total_pages = len(pdf_reader.pages)
    
    # List to store the split PDFs as BytesIO objects
    split_pdfs = []
    
    # Iterate over each group of page indices
    for group in page_groups:
        # Create a new PDF writer for this group
        pdf_writer = PdfWriter()
        
        # Add the pages in the current group to the writer
        for page_idx in group:
            if 0 <= page_idx < total_pages:
                pdf_writer.add_page(pdf_reader.pages[page_idx])
            else:
                print(f"Warning: Page index {page_idx} is out of range (0 to {total_pages-1}). Skipping.")
        
        # Write the group of pages to a new BytesIO object
        output_bytes_io = BytesIO()
        pdf_writer.write(output_bytes_io)
        output_bytes_io.seek(0)  # Reset the pointer to the start of the stream
        split_pdfs.append(output_bytes_io)
    
    return split_pdfs


def split_excel_sheets_to_bytes(input_bytes):
    """Split an Excel workbook into separate single-sheet Excel files in memory.

    Each sheet is re-written as an independent .xlsx file using xlsxwriter via
    pandas ExcelWriter. This is used to process sheets individually during
    Excel extraction workflows.

    Args:
        input_bytes: BytesIO object containing the source Excel file data.

    Returns:
        A dict mapping sheet names (str) to BytesIO objects, each containing
        a standalone .xlsx file with that single sheet.

    Raises:
        Exception: Propagates any pandas or xlsxwriter errors encountered
            during reading or writing.
    """
    sheet_bytes_dict = {}
    try:
        # Read the Excel file from the BytesIO object
        input_bytes.seek(0)  # Ensure pointer is at the start
        excel_file = pd.ExcelFile(input_bytes)
        print("Sheet names found:", excel_file.sheet_names)
        
        # Iterate through each sheet in the Excel file
        for sheet_name in excel_file.sheet_names:
            input_bytes.seek(0)  # Reset pointer before reading each sheet
            df = pd.read_excel(input_bytes, sheet_name=sheet_name)
            print(f"Read data for sheet: {sheet_name}, shape: {df.shape}")
            
            # Create a BytesIO object to store the Excel data in memory
            output = BytesIO()
            try:
                # Write the DataFrame to the BytesIO object as an Excel file using xlsxwriter
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name=sheet_name)
                print(f"Written data for sheet: {sheet_name} to BytesIO")
            except Exception as e:
                print(f"Error writing to BytesIO for sheet {sheet_name}: {e}")
                raise
            
            output.seek(0)  # Reset pointer to start after writing
            sheet_bytes_dict[sheet_name] = output
    
    except Exception as e:
        print(f"Error processing input Excel file: {e}")
        raise
    
    return sheet_bytes_dict


def extract_filename_from_url(url: str) -> str:
    """Extract the base filename (without extension) from a URL.

    Parses the URL path component to retrieve the last segment and strips
    any query parameters and the file extension.

    Args:
        url: The URL containing the file path (e.g. an Azure Blob Storage URL).

    Returns:
        The base filename without extension, or "default_file_name" if
        extraction fails.
    """
    try:
        # Parse the URL to get the path component
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Split the path by '/' and get the last segment (filename with extension)
        filename_with_ext = path.split('/')[-1]
        
        # Remove query parameters if they are part of the last segment (e.g., after '?')
        filename_with_ext = filename_with_ext.split('?')[0]
        
        # Extract the base filename without extension
        base_filename = os.path.splitext(filename_with_ext)[0]
        
        return base_filename
    except Exception as e:
        print(f"Error extracting filename from URL: {e}")
        return "default_file_name"