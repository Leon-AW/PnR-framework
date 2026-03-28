"""
Structure-Aware Document Chunker
================================

Chunks QM documents while preserving structural elements like tables,
lists, and section hierarchies.

Features:
- PATH extraction from QM documents
- Section breadcrumb tracking
- Atomic table handling (keeps tables together)
- List grouping for coherent context
- Overlap between chunks for continuity
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StructuredChunkConfig:
    """Configuration for structure-aware document chunking.

    Attributes:
        max_chunk_tokens: Maximum tokens per chunk (default: 750)
        table_max_tokens: Keep tables atomic up to this size (default: 1500)
        list_max_tokens: Keep lists together up to this size (default: 500)
        overlap_tokens: Overlap between chunks (default: 50)
        include_breadcrumb: Include section breadcrumb in chunk (default: True)
        include_path: Include document path in chunk (default: True)
    """
    max_chunk_tokens: int = 750
    table_max_tokens: int = 1500
    list_max_tokens: int = 500
    overlap_tokens: int = 50
    include_breadcrumb: bool = True
    include_path: bool = True


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StructuredChunk:
    """A structured document chunk with metadata.

    Attributes:
        content: The chunk text content
        path: Document path (e.g., "QM/DE/LKR/Prüfstelle/...")
        section_breadcrumb: Section hierarchy (e.g., "Metallografie > Prüfverfahren > Mikrohärte")
        content_type: Type of content in chunk
        source_file: Path to source document
        chunk_index: Index within document
        start_char: Starting character position
        end_char: Ending character position
        parent_section: Parent section title (for parent-child RAG)
        metadata: Additional metadata
    """
    content: str
    path: str
    section_breadcrumb: str
    content_type: Literal["paragraph", "table", "image", "list", "mixed"]
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    parent_section: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def format_with_context(self) -> str:
        """Format chunk with document path and section context."""
        parts = []
        if self.path:
            parts.append(f"[Document: {self.path}]")
        if self.section_breadcrumb:
            parts.append(f"[Section: {self.section_breadcrumb}]")
        if parts:
            parts.append("")
        parts.append(self.content)
        return "\n".join(parts)


# =============================================================================
# Patterns for QM Document Structure
# =============================================================================

# Path declaration at start of document
PATH_PATTERN = re.compile(r'^Path:\s*["\']?(.+?)["\']?\s*$', re.IGNORECASE | re.MULTILINE)

# Markdown headers (# to ######)
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+\*{0,2}(.+?)\*{0,2}\s*$', re.MULTILINE)

# Table rows (markdown tables)
TABLE_ROW_PATTERN = re.compile(r'^\|.*\|', re.MULTILINE)

# Table separator (---|---|---)
TABLE_SEP_PATTERN = re.compile(r'^\|[\s\-:]+\|', re.MULTILINE)

# List items (- , * , 1. , etc.)
LIST_ITEM_PATTERN = re.compile(r'^(\s*)([-*+]|\d+\.)\s+', re.MULTILINE)

# Image references
IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


# =============================================================================
# Content Block Classes
# =============================================================================

@dataclass
class ContentBlock:
    """A block of content with type information."""
    content: str
    block_type: Literal["paragraph", "table", "list", "header", "image"]
    start_char: int
    end_char: int
    header_level: int = 0
    header_text: str = ""


# =============================================================================
# Structure-Aware Chunker
# =============================================================================

class StructureAwareChunker:
    """Chunks QM documents while preserving structure.

    Example:
        ```python
        config = StructuredChunkConfig(max_chunk_tokens=750)
        chunker = StructureAwareChunker(config)

        chunks = chunker.chunk_document("path/to/qm_doc.md")

        for chunk in chunks:
            print(f"[{chunk.content_type}] {chunk.section_breadcrumb}")
            print(chunk.format_with_context())
        ```
    """

    def __init__(self, config: Optional[StructuredChunkConfig] = None):
        """Initialize the chunker.

        Args:
            config: Chunking configuration (uses defaults if None)
        """
        self.config = config or StructuredChunkConfig()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(text) // 4

    def _extract_path(self, content: str) -> tuple[str, str]:
        """Extract document path from content.

        Args:
            content: Full document content

        Returns:
            Tuple of (path, content_without_path_line)
        """
        match = PATH_PATTERN.search(content)
        if match:
            path = match.group(1).strip()
            # Remove the path line from content
            content_clean = content[:match.start()] + content[match.end():]
            return path, content_clean.strip()
        return "", content

    def _parse_blocks(self, content: str) -> list[ContentBlock]:
        """Parse content into typed blocks.

        Args:
            content: Document content (without path line)

        Returns:
            List of ContentBlock objects
        """
        blocks = []
        lines = content.split("\n")

        i = 0
        char_pos = 0

        while i < len(lines):
            line = lines[i]
            line_start = char_pos

            # Check for header
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2).strip()
                blocks.append(ContentBlock(
                    content=line,
                    block_type="header",
                    start_char=line_start,
                    end_char=line_start + len(line),
                    header_level=level,
                    header_text=text,
                ))
                char_pos += len(line) + 1  # +1 for newline
                i += 1
                continue

            # Check for table start
            if TABLE_ROW_PATTERN.match(line):
                table_lines = [line]
                table_start = line_start
                i += 1
                char_pos += len(line) + 1

                # Collect all table rows
                while i < len(lines) and (TABLE_ROW_PATTERN.match(lines[i]) or TABLE_SEP_PATTERN.match(lines[i]) or lines[i].strip() == ""):
                    if lines[i].strip() == "":
                        # Check if next line continues table
                        if i + 1 < len(lines) and TABLE_ROW_PATTERN.match(lines[i + 1]):
                            table_lines.append(lines[i])
                            char_pos += len(lines[i]) + 1
                            i += 1
                        else:
                            break
                    else:
                        table_lines.append(lines[i])
                        char_pos += len(lines[i]) + 1
                        i += 1

                blocks.append(ContentBlock(
                    content="\n".join(table_lines),
                    block_type="table",
                    start_char=table_start,
                    end_char=char_pos - 1,
                ))
                continue

            # Check for list
            if LIST_ITEM_PATTERN.match(line):
                list_lines = [line]
                list_start = line_start
                base_indent = len(line) - len(line.lstrip())
                i += 1
                char_pos += len(line) + 1

                # Collect list items (including nested)
                while i < len(lines):
                    curr_line = lines[i]
                    curr_indent = len(curr_line) - len(curr_line.lstrip())

                    # Continue if: list item, continuation line, or blank line in list
                    if LIST_ITEM_PATTERN.match(curr_line):
                        list_lines.append(curr_line)
                        char_pos += len(curr_line) + 1
                        i += 1
                    elif curr_indent > base_indent and curr_line.strip():
                        # Continuation of list item
                        list_lines.append(curr_line)
                        char_pos += len(curr_line) + 1
                        i += 1
                    elif curr_line.strip() == "" and i + 1 < len(lines) and LIST_ITEM_PATTERN.match(lines[i + 1]):
                        # Blank line within list
                        list_lines.append(curr_line)
                        char_pos += len(curr_line) + 1
                        i += 1
                    else:
                        break

                blocks.append(ContentBlock(
                    content="\n".join(list_lines),
                    block_type="list",
                    start_char=list_start,
                    end_char=char_pos - 1,
                ))
                continue

            # Check for image
            if IMAGE_PATTERN.search(line):
                blocks.append(ContentBlock(
                    content=line,
                    block_type="image",
                    start_char=line_start,
                    end_char=line_start + len(line),
                ))
                char_pos += len(line) + 1
                i += 1
                continue

            # Regular paragraph - collect until next structural element
            para_lines = [line]
            para_start = line_start
            i += 1
            char_pos += len(line) + 1

            while i < len(lines):
                curr_line = lines[i]

                # Stop at structural elements
                if (HEADER_PATTERN.match(curr_line) or
                    TABLE_ROW_PATTERN.match(curr_line) or
                    LIST_ITEM_PATTERN.match(curr_line) or
                    IMAGE_PATTERN.search(curr_line)):
                    break

                # Stop at double blank line (section break)
                if curr_line.strip() == "" and para_lines and para_lines[-1].strip() == "":
                    break

                para_lines.append(curr_line)
                char_pos += len(curr_line) + 1
                i += 1

            para_content = "\n".join(para_lines).strip()
            if para_content:
                blocks.append(ContentBlock(
                    content=para_content,
                    block_type="paragraph",
                    start_char=para_start,
                    end_char=char_pos - 1,
                ))

        return blocks

    def _build_breadcrumb(self, headers: list[tuple[int, str]]) -> str:
        """Build section breadcrumb from header stack.

        Args:
            headers: List of (level, text) tuples

        Returns:
            Breadcrumb string like "Section > Subsection > Subsubsection"
        """
        if not headers:
            return ""
        return " > ".join(text for _, text in headers)

    def _update_header_stack(
        self,
        stack: list[tuple[int, str]],
        level: int,
        text: str
    ) -> list[tuple[int, str]]:
        """Update header stack when encountering a new header.

        Args:
            stack: Current header stack
            level: New header level (1-6)
            text: New header text

        Returns:
            Updated header stack
        """
        # Remove headers at same or lower level
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        return stack

    def chunk_document(self, file_path: str | Path) -> list[StructuredChunk]:
        """Chunk a QM document with structure awareness.

        Args:
            file_path: Path to document

        Returns:
            List of StructuredChunk objects
        """
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract document path
        doc_path, content = self._extract_path(content)

        # Parse into blocks
        blocks = self._parse_blocks(content)

        if not blocks:
            return []

        # Build chunks
        chunks = []
        header_stack: list[tuple[int, str]] = []
        current_chunk_blocks: list[ContentBlock] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._estimate_tokens(block.content)

            # Handle headers - update stack but don't create standalone chunks
            if block.block_type == "header":
                header_stack = self._update_header_stack(
                    header_stack, block.header_level, block.header_text
                )
                # Add header to current chunk if there's content
                if current_chunk_blocks:
                    current_chunk_blocks.append(block)
                    current_tokens += block_tokens
                continue

            # Handle tables - keep atomic if possible
            if block.block_type == "table":
                if block_tokens <= self.config.table_max_tokens:
                    # Flush current chunk if table doesn't fit
                    if current_tokens + block_tokens > self.config.max_chunk_tokens and current_chunk_blocks:
                        chunks.append(self._create_chunk(
                            current_chunk_blocks,
                            doc_path,
                            header_stack,
                            str(path),
                            len(chunks),
                        ))
                        current_chunk_blocks = []
                        current_tokens = 0

                    current_chunk_blocks.append(block)
                    current_tokens += block_tokens
                else:
                    # Table too large - flush and add as separate chunk
                    if current_chunk_blocks:
                        chunks.append(self._create_chunk(
                            current_chunk_blocks,
                            doc_path,
                            header_stack,
                            str(path),
                            len(chunks),
                        ))
                        current_chunk_blocks = []
                        current_tokens = 0

                    chunks.append(self._create_chunk(
                        [block],
                        doc_path,
                        header_stack,
                        str(path),
                        len(chunks),
                    ))
                continue

            # Handle lists - keep together if possible
            if block.block_type == "list":
                if block_tokens <= self.config.list_max_tokens:
                    if current_tokens + block_tokens > self.config.max_chunk_tokens and current_chunk_blocks:
                        chunks.append(self._create_chunk(
                            current_chunk_blocks,
                            doc_path,
                            header_stack,
                            str(path),
                            len(chunks),
                        ))
                        current_chunk_blocks = []
                        current_tokens = 0

                    current_chunk_blocks.append(block)
                    current_tokens += block_tokens
                else:
                    # List too large - need to split
                    if current_chunk_blocks:
                        chunks.append(self._create_chunk(
                            current_chunk_blocks,
                            doc_path,
                            header_stack,
                            str(path),
                            len(chunks),
                        ))
                        current_chunk_blocks = []
                        current_tokens = 0

                    # Split list by items
                    list_chunks = self._split_large_list(block, doc_path, header_stack, str(path), len(chunks))
                    chunks.extend(list_chunks)
                continue

            # Handle paragraphs and other content
            if current_tokens + block_tokens > self.config.max_chunk_tokens:
                if current_chunk_blocks:
                    chunks.append(self._create_chunk(
                        current_chunk_blocks,
                        doc_path,
                        header_stack,
                        str(path),
                        len(chunks),
                    ))
                    current_chunk_blocks = []
                    current_tokens = 0

            current_chunk_blocks.append(block)
            current_tokens += block_tokens

        # Flush remaining content
        if current_chunk_blocks:
            chunks.append(self._create_chunk(
                current_chunk_blocks,
                doc_path,
                header_stack,
                str(path),
                len(chunks),
            ))

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        logger.debug(f"Created {len(chunks)} structured chunks from {path}")
        return chunks

    def _create_chunk(
        self,
        blocks: list[ContentBlock],
        doc_path: str,
        header_stack: list[tuple[int, str]],
        source_file: str,
        chunk_index: int,
    ) -> StructuredChunk:
        """Create a StructuredChunk from content blocks.

        Args:
            blocks: List of content blocks
            doc_path: Document path string
            header_stack: Current header stack for breadcrumb
            source_file: Source file path
            chunk_index: Chunk index

        Returns:
            StructuredChunk object
        """
        # Combine block content
        content_parts = []
        for block in blocks:
            content_parts.append(block.content)
        content = "\n\n".join(content_parts)

        # Determine content type
        block_types = {b.block_type for b in blocks if b.block_type != "header"}
        if len(block_types) == 1:
            content_type = block_types.pop()
        elif block_types:
            content_type = "mixed"
        else:
            content_type = "paragraph"

        # Build breadcrumb
        breadcrumb = self._build_breadcrumb(header_stack)

        # Get parent section
        parent_section = header_stack[-1][1] if header_stack else None

        # Calculate char positions
        start_char = min(b.start_char for b in blocks)
        end_char = max(b.end_char for b in blocks)

        return StructuredChunk(
            content=content,
            path=doc_path if self.config.include_path else "",
            section_breadcrumb=breadcrumb if self.config.include_breadcrumb else "",
            content_type=content_type,
            source_file=source_file,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            parent_section=parent_section,
        )

    def _split_large_list(
        self,
        block: ContentBlock,
        doc_path: str,
        header_stack: list[tuple[int, str]],
        source_file: str,
        start_index: int,
    ) -> list[StructuredChunk]:
        """Split a large list into multiple chunks.

        Args:
            block: The list content block
            doc_path: Document path
            header_stack: Current header stack
            source_file: Source file path
            start_index: Starting chunk index

        Returns:
            List of StructuredChunk objects
        """
        chunks = []
        lines = block.content.split("\n")

        current_items: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is a list item start
            if LIST_ITEM_PATTERN.match(line):
                item_lines = [line]
                base_indent = len(line) - len(line.lstrip())
                i += 1

                # Collect continuation lines
                while i < len(lines):
                    curr_line = lines[i]
                    curr_indent = len(curr_line) - len(curr_line.lstrip())

                    if LIST_ITEM_PATTERN.match(curr_line) and curr_indent <= base_indent:
                        break
                    elif curr_indent > base_indent or curr_line.strip() == "":
                        item_lines.append(curr_line)
                        i += 1
                    else:
                        break

                item_text = "\n".join(item_lines)
                item_tokens = self._estimate_tokens(item_text)

                if current_tokens + item_tokens > self.config.max_chunk_tokens and current_items:
                    # Create chunk from current items
                    chunks.append(StructuredChunk(
                        content="\n".join(current_items),
                        path=doc_path if self.config.include_path else "",
                        section_breadcrumb=self._build_breadcrumb(header_stack) if self.config.include_breadcrumb else "",
                        content_type="list",
                        source_file=source_file,
                        chunk_index=start_index + len(chunks),
                        start_char=block.start_char,
                        end_char=block.end_char,
                        parent_section=header_stack[-1][1] if header_stack else None,
                    ))
                    current_items = []
                    current_tokens = 0

                current_items.append(item_text)
                current_tokens += item_tokens
            else:
                i += 1

        # Flush remaining items
        if current_items:
            chunks.append(StructuredChunk(
                content="\n".join(current_items),
                path=doc_path if self.config.include_path else "",
                section_breadcrumb=self._build_breadcrumb(header_stack) if self.config.include_breadcrumb else "",
                content_type="list",
                source_file=source_file,
                chunk_index=start_index + len(chunks),
                start_char=block.start_char,
                end_char=block.end_char,
                parent_section=header_stack[-1][1] if header_stack else None,
            ))

        return chunks

    def chunk_documents(self, file_paths: list[str | Path]) -> list[StructuredChunk]:
        """Chunk multiple documents.

        Args:
            file_paths: List of paths to documents

        Returns:
            List of all StructuredChunk objects
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = self.chunk_document(file_path)
            all_chunks.extend(chunks)
        return all_chunks
