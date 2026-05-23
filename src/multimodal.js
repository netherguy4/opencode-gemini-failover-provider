import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { config, SUPPORTED_IMAGE_MIMES, TEXT_MIMES, EXTRACTABLE_MIMES } from "./config.js";

export class MultimodalError extends Error {
  constructor(message, { status = 400, code, details } = {}) {
    super(message);
    this.name = "MultimodalError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

// ---- Data URL helpers ----

export function parseDataUrl(url) {
  if (typeof url !== "string") return null;
  const match = url.match(/^data:([^;]*);?([^,]*)?,(.*)$/s);
  if (!match) return null;
  const mimeType = match[1] || null;
  const isBase64 = match[2] === "base64";
  const data = match[3];
  return { mimeType, isBase64, data, raw: url };
}

export function estimateBytesFromBase64(b64) {
  return Math.floor((String(b64 || "").length * 3) / 4);
}

export function estimateDataUrlBytes(url) {
  const parsed = parseDataUrl(url);
  if (!parsed) return 0;
  return estimateBytesFromBase64(parsed.data);
}

export function validateBase64(b64) {
  // Basic base64 validation
  return /^[A-Za-z0-9+/]*={0,2}$/.test(String(b64 || ""));
}

// ---- Image handling ----

const MIME_TO_EXT = {
  "image/png": "png",
  "image/jpeg": "jpeg",
  "image/webp": "webp",
  "image/gif": "gif",
};

export function normalizeImagePart(part) {
  // Collect URL and optional metadata
  let url = null;
  let base64 = null;
  let mimeType = null;

  // { type: "image_url", image_url: { url: "data:..." } }
  if (part.image_url) {
    if (typeof part.image_url === "string") {
      url = part.image_url;
    } else if (typeof part.image_url === "object") {
      url = part.image_url.url || null;
    }
  }

  // { type: "input_image", base64: "...", mime_type: "..." }
  // { type: "input_image", url: "data:..." }
  if (!url && part.base64) {
    base64 = part.base64;
    mimeType = part.mime_type || null;
  }
  if (!url && part.url) {
    url = part.url;
  }

  if (!url && !base64) {
    return { kind: "unsupported", reason: "image part has no usable image data" };
  }

  // If we have url, parse it
  if (url) {
    const parsed = parseDataUrl(url);
    if (!parsed) {
      return { kind: "unsupported", reason: "image URL is not a valid data URL" };
    }
    base64 = parsed.data;
    if (!mimeType) mimeType = parsed.mimeType;
    if (!validateBase64(base64)) {
      return { kind: "unsupported", reason: "image contains invalid base64 data" };
    }
  }

  // Build data URL
  let dataUrl;
  if (!url) {
    const mime = mimeType || "image/png";
    dataUrl = `data:${mime};base64,${base64}`;
  } else {
    dataUrl = url;
  }

  // Check MIME
  const effectiveMime = mimeType || extractMimeFromPart(part) || "image/png";
  if (!SUPPORTED_IMAGE_MIMES.has(effectiveMime)) {
    return {
      kind: "unsupported",
      reason: `unsupported image MIME type: ${effectiveMime}. Supported: ${[...SUPPORTED_IMAGE_MIMES].join(", ")}`,
      mimeType: effectiveMime,
    };
  }

  // Check size
  const bytes = estimateDataUrlBytes(dataUrl);
  if (bytes > config.maxImageBytes) {
    return {
      kind: "unsupported",
      reason: `image too large (${formatBytes(bytes)} > ${formatBytes(config.maxImageBytes)} max)`,
      mimeType: effectiveMime,
    };
  }

  const ext = MIME_TO_EXT[effectiveMime] || "png";
  return {
    kind: "image",
    mimeType: effectiveMime,
    dataUrl,
    base64,
    filename: part?.file?.filename || part?.filename || `image.${ext}`,
    bytes,
  };
}

// ---- File handling ----

export function normalizeFilePart(part) {
  let fileData = null;
  let base64 = null;
  let text = null;
  let filename = null;
  let mimeType = null;

  // { type: "file", file: { filename: "...", file_data: "data:..." } }
  // { type: "input_file", filename: "...", file_data: "data:...", base64: "...", mime_type: "...", text: "..." }
  if (part.file && typeof part.file === "object") {
    filename = part.file.filename || part.filename || null;
    fileData = part.file.file_data || null;
  }
  if (part.file_data) fileData = part.file_data;
  if (part.base64) base64 = part.base64;
  if (part.text) text = part.text;
  if (!filename) filename = part.filename || null;
  if (!mimeType) mimeType = part.mime_type || part.file?.mime_type || null;

  // Parse file_data data URL
  if (fileData && !base64 && !text) {
    const parsed = parseDataUrl(fileData);
    if (parsed) {
      base64 = parsed.data;
      if (!mimeType) mimeType = parsed.mimeType;
    } else {
      // Could be raw base64
      base64 = fileData;
    }
  }

  const effectiveMime = mimeType || guessMimeFromFilename(filename) || "application/octet-stream";

  // Check size
  let bytes = 0;
  if (base64) {
    bytes = estimateBytesFromBase64(base64);
    if (bytes > config.maxFileBytes) {
      return {
        kind: "unsupported",
        reason: `file too large (${formatBytes(bytes)} > ${formatBytes(config.maxFileBytes)} max)`,
        mimeType: effectiveMime,
        filename,
      };
    }
  } else if (text) {
    bytes = Buffer.byteLength(text, "utf8");
  }

  return {
    kind: "file",
    mimeType: effectiveMime,
    filename: filename || "file.bin",
    base64,
    text,
    bytes,
  };
}

// ---- Text extraction ----

export async function extractFileText(normalizedFile) {
  const { mimeType, base64, text, filename } = normalizedFile;

  // Already text
  if (text) {
    return truncateText(text, config.fileTextMaxChars, filename);
  }

  if (!base64) {
    throw new MultimodalError(`No file content for ${filename}`, {
      code: "empty_file",
      details: { filename, mimeType },
    });
  }

  const buffer = Buffer.from(base64, "base64");

  // Check if it's a text MIME - decode directly
  if (TEXT_MIMES.has(mimeType)) {
    const str = buffer.toString("utf8");
    return processTextContent(str, mimeType, filename);
  }

  // Check if it's extractable
  if (!EXTRACTABLE_MIMES.has(mimeType)) {
    throw new MultimodalError(
      `Unsupported attachment type: ${mimeType} for ${filename}. Supported: images, pdf, docx, xlsx, txt, md, json, csv, html.`,
      { code: "unsupported_attachment_type", details: { filename, mimeType } }
    );
  }

  // HTML - strip tags
  if (mimeType === "text/html") {
    const str = buffer.toString("utf8");
    return processTextContent(str, mimeType, filename);
  }

  // PDF
  if (mimeType === "application/pdf") {
    return extractPdfText(buffer, filename);
  }

  // DOCX
  if (mimeType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
    return extractDocxText(buffer, filename);
  }

  // XLSX
  if (mimeType === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") {
    return extractXlsxText(buffer, filename);
  }

  throw new MultimodalError(`No extraction handler for: ${mimeType}`, {
    code: "file_extraction_unsupported",
    details: { filename, mimeType },
  });
}

async function extractPdfText(buffer, filename) {
  try {
    const pdfParse = (await import("pdf-parse")).default;
    const data = await pdfParse(buffer);
    const text = data.text || "";
    const formatted = `[Attached file: ${filename}]\nMIME type: application/pdf\nExtracted text:\n---\n${text}\n---`;
    return truncateText(formatted, config.fileTextMaxChars, filename);
  } catch (e) {
    throw new MultimodalError(`Failed to extract text from PDF: ${e.message}`, {
      code: "file_extraction_failed",
      details: { filename, mimeType: "application/pdf" },
    });
  }
}

async function extractDocxText(buffer, filename) {
  try {
    const mammoth = (await import("mammoth")).default;
    // Convert buffer to something mammoth can use
    const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const result = await mammoth.extractRawText({ arrayBuffer });
    const text = result.value || "";
    const formatted = `[Attached file: ${filename}]\nMIME type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\nExtracted text:\n---\n${text}\n---`;
    return truncateText(formatted, config.fileTextMaxChars, filename);
  } catch (e) {
    throw new MultimodalError(`Failed to extract text from DOCX: ${e.message}`, {
      code: "file_extraction_failed",
      details: { filename, mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" },
    });
  }
}

async function extractXlsxText(buffer, filename) {
  try {
    const XLSX = (await import("xlsx")).default;
    const workbook = XLSX.read(buffer, { type: "buffer" });
    const sheetNames = workbook.SheetNames;
    let result = `[Attached spreadsheet: ${filename}]\n`;

    for (const name of sheetNames) {
      const sheet = workbook.Sheets[name];
      const csvText = XLSX.utils.sheet_to_csv(sheet);
      // Convert CSV to a readable summary (truncate to avoid flooding context)
      const lines = csvText.split("\n").filter(Boolean);
      const maxRows = Math.min(lines.length, 100);
      const preview = lines.slice(0, maxRows).join("\n");
      const truncated = lines.length > maxRows ? `\n[... ${lines.length - maxRows} more rows omitted]` : "";

      result += `\nSheet: ${name}\n`;
      // Format as a simple table if reasonable
      if (preview.includes(",")) {
        result += `| ${preview.split("\n").join(" |\n| ")} |\n`;
      } else {
        result += `${preview}\n`;
      }
      result += truncated;
    }

    return truncateText(result, config.fileTextMaxChars, filename);
  } catch (e) {
    throw new MultimodalError(`Failed to extract text from XLSX: ${e.message}`, {
      code: "file_extraction_failed",
      details: { filename, mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" },
    });
  }
}

// ---- Text content processing ----

function processTextContent(str, mimeType, filename) {
  let processed = str;

  if (mimeType === "text/html") {
    processed = stripHtml(str);
  }

  const formatted = `[Attached file: ${filename}]\nMIME type: ${mimeType}\n---\n${processed}\n---`;
  return truncateText(formatted, config.fileTextMaxChars, filename);
}

function stripHtml(html) {
  return html
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function truncateText(text, maxChars, filename) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + `\n[File text truncated after ${maxChars} characters]`;
}

// ---- MIME type helpers ----

export function guessMimeFromFilename(filename) {
  if (!filename) return null;
  const ext = String(filename).split(".").pop()?.toLowerCase();
  const map = {
    txt: "text/plain",
    md: "text/markdown",
    json: "application/json",
    csv: "text/csv",
    html: "text/html",
    htm: "text/html",
    pdf: "application/pdf",
    docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    doc: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    xls: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    webp: "image/webp",
    gif: "image/gif",
  };
  return map[ext] || null;
}

export function extractMimeFromPart(part) {
  if (!part || typeof part !== "object") return null;
  if (part.mime_type) return part.mime_type;
  if (part.image_url?.detail === "auto") {
    // Never gives us MIME info; fallback to filename
  }
  if (part.file?.mime_type) return part.file.mime_type;
  if (part.filename) return guessMimeFromFilename(part.filename);
  return null;
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1048576).toFixed(1)}MB`;
}

// ---- Content normalization ----

export function normalizeContent(content, _options = {}) {
  const parts = [];
  const warnings = [];

  if (content == null) {
    return { parts: [{ kind: "text", text: "" }], warnings };
  }

  if (typeof content === "string") {
    return { parts: [{ kind: "text", text: content }], warnings };
  }

  if (!Array.isArray(content)) {
    return { parts: [{ kind: "text", text: String(content) }], warnings };
  }

  let totalAttachmentBytes = 0;

  for (const part of content) {
    if (typeof part === "string") {
      parts.push({ kind: "text", text: part });
      continue;
    }

    if (!part || typeof part !== "object") continue;

    // Text parts
    if (part.type === "text" || part.type === "input_text") {
      parts.push({ kind: "text", text: part.text || "" });
      continue;
    }

    // Image parts
    if (part.type === "image_url" || part.type === "input_image") {
      const normalized = normalizeImagePart(part);
      if (normalized.kind === "image") {
        totalAttachmentBytes += normalized.bytes || 0;
        if (totalAttachmentBytes > config.maxTotalAttachmentBytes) {
          warnings.push(`total attachment size exceeds ${formatBytes(config.maxTotalAttachmentBytes)}`);
          continue;
        }
      }
      parts.push(normalized);
      continue;
    }

    // File parts
    if (part.type === "file" || part.type === "input_file") {
      const normalized = normalizeFilePart(part);
      if (normalized.kind === "file") {
        totalAttachmentBytes += normalized.bytes || 0;
        if (totalAttachmentBytes > config.maxTotalAttachmentBytes) {
          warnings.push(`total attachment size exceeds ${formatBytes(config.maxTotalAttachmentBytes)}`);
          continue;
        }
      }
      parts.push(normalized);
      continue;
    }

    // Unknown part type
    const mimeGuess = extractMimeFromPart(part);
    if (mimeGuess && SUPPORTED_IMAGE_MIMES.has(mimeGuess)) {
      // Re-try as image
      const normalized = normalizeImagePart({ ...part, type: "image_url", mime_type: mimeGuess });
      parts.push(normalized);
      continue;
    }

    warnings.push(`unsupported content part type: ${part.type || "unknown"}`);
    parts.push({ kind: "unsupported", reason: `unsupported content part: ${part.type || "unknown"}` });
  }

  return { parts, warnings };
}

// ---- Build upstream OpenAI-compatible content ----

export async function buildUpstreamContent(content, enableFileExtraction = true) {
  const { parts, warnings } = normalizeContent(content);

  const hasFiles = parts.some((p) => p.kind === "file");
  let extractedTexts = [];

  if (hasFiles && enableFileExtraction) {
    for (const part of parts) {
      if (part.kind === "file") {
        try {
          const text = await extractFileText(part);
          extractedTexts.push(text);
        } catch (e) {
          if (e instanceof MultimodalError) throw e;
          throw new MultimodalError(`Failed to process file: ${e.message}`, {
            code: "file_extraction_failed",
            details: { filename: part.filename, mimeType: part.mimeType },
          });
        }
      }
    }
  }

  // Check for unsupported parts that should be errors
  for (const part of parts) {
    if (part.kind === "unsupported") {
      throw new MultimodalError(part.reason, {
        code: "unsupported_attachment_type",
        details: part,
      });
    }
  }

  // Build OpenAI content array
  const openAiContent = [];
  let hasAnyImage = false;

  for (const part of parts) {
    if (part.kind === "text") {
      openAiContent.push({ type: "text", text: part.text });
    } else if (part.kind === "image") {
      hasAnyImage = true;
      openAiContent.push({
        type: "image_url",
        image_url: { url: part.dataUrl },
      });
    }
    // File parts are converted to text and injected below
  }

  // Inject extracted file texts into the content
  if (extractedTexts.length > 0) {
    // Append file texts at the end as text blocks
    for (const fileText of extractedTexts) {
      openAiContent.push({ type: "text", text: fileText });
    }
  }

  return { openAiContent, warnings, hasAnyImage };
}

// ---- Ollama images[] helper ----

export function ollamaImagesToParts(images) {
  if (!Array.isArray(images) || images.length === 0) return [];

  return images
    .map((img) => {
      if (typeof img !== "string" || !img) return null;
      const url = img.startsWith("data:") ? img : `data:image/png;base64,${img}`;
      const parsed = parseDataUrl(url);
      const mimeType = parsed?.mimeType || "image/png";
      return {
        type: "image_url",
        image_url: { url },
        _mimeType: mimeType,
      };
    })
    .filter(Boolean);
}
