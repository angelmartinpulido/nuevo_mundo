"""
Source-specific adapters for data collection
"""

from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from .collector_core import DataSource
import logging
import json
import os
import aiofiles
from urllib.parse import urlparse
import re


class WebDataSource(DataSource):
    """Collects data from any web source using intelligent parsing"""

    def __init__(self, url: str):
        self.url = url
        self.session: Optional[aiohttp.ClientSession] = None
        self.parser = WebContentParser()

    async def connect(self) -> bool:
        """Establish connection to web source"""
        try:
            self.session = aiohttp.ClientSession()
            return True
        except Exception as e:
            logging.error(f"Failed to create web session: {str(e)}")
            return False

    async def collect(self) -> Dict[str, Any]:
        """Collect and parse web content"""
        try:
            async with self.session.get(self.url) as response:
                content = await response.text()
                return await self.parser.parse(content, self.url)
        finally:
            if self.session:
                await self.session.close()

    async def validate(self, data: Dict[str, Any]) -> bool:
        """Validate collected web data"""
        return bool(data and isinstance(data, dict))


class GitHubDataSource(DataSource):
    """Intelligent GitHub data collector"""

    def __init__(self, repo_url: str, token: Optional[str] = None):
        self.repo_url = repo_url
        self.api_url = self._convert_to_api_url(repo_url)
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    def _convert_to_api_url(self, repo_url: str) -> str:
        """Convert GitHub web URL to API URL"""
        parts = repo_url.split("github.com/")
        if len(parts) == 2:
            return f"https://api.github.com/repos/{parts[1]}"
        return repo_url

    async def connect(self) -> bool:
        """Connect to GitHub API"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        try:
            self.session = aiohttp.ClientSession(headers=headers)
            return True
        except Exception as e:
            logging.error(f"Failed to connect to GitHub: {str(e)}")
            return False

    async def collect(self) -> Dict[str, Any]:
        """Collect repository data intelligently"""
        try:
            data = {}

            # Collect basic repo info
            async with self.session.get(self.api_url) as response:
                data["repo_info"] = await response.json()

            # Collect code structure
            async with self.session.get(
                f"{self.api_url}/git/trees/master?recursive=1"
            ) as response:
                data["structure"] = await response.json()

            # Collect recent commits
            async with self.session.get(f"{self.api_url}/commits") as response:
                data["commits"] = await response.json()

            return data
        finally:
            if self.session:
                await self.session.close()

    async def validate(self, data: Dict[str, Any]) -> bool:
        """Validate GitHub data"""
        return all(key in data for key in ["repo_info", "structure", "commits"])


class LocalNodeDataSource(DataSource):
    """Collects data from local node"""

    def __init__(self, paths: List[str]):
        self.paths = paths
        self.file_patterns = ["*.json", "*.txt", "*.md", "*.py", "*.js", "*.csv"]

    async def connect(self) -> bool:
        """Verify access to local paths"""
        return all(os.path.exists(path) for path in self.paths)

    async def collect(self) -> Dict[str, Any]:
        """Collect local node data"""
        data = {}

        for path in self.paths:
            data[path] = await self._collect_path_data(path)

        return data

    async def _collect_path_data(self, path: str) -> Dict[str, Any]:
        """Collect data from a specific path"""
        path_data = {}

        if os.path.isfile(path):
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
                path_data["content"] = content
                path_data["type"] = "file"

        elif os.path.isdir(path):
            path_data["type"] = "directory"
            path_data["contents"] = {}

            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(
                        file.endswith(pat.replace("*", ""))
                        for pat in self.file_patterns
                    ):
                        file_path = os.path.join(root, file)
                        try:
                            async with aiofiles.open(file_path, "r") as f:
                                content = await f.read()
                                path_data["contents"][file_path] = content
                        except Exception as e:
                            logging.warning(f"Could not read {file_path}: {str(e)}")

        return path_data

    async def validate(self, data: Dict[str, Any]) -> bool:
        """Validate local node data"""
        return isinstance(data, dict) and bool(data)


class WebContentParser:
    """Intelligent web content parser"""

    async def parse(self, content: str, url: str) -> Dict[str, Any]:
        """Parse web content based on URL and content type"""
        parsed_data = {
            "url": url,
            "content_type": self._detect_content_type(content),
            "structured_data": {},
            "metadata": {},
        }

        # Extract structured data
        parsed_data["structured_data"] = await self._extract_structured_data(content)

        # Extract metadata
        parsed_data["metadata"] = self._extract_metadata(content)

        return parsed_data

    def _detect_content_type(self, content: str) -> str:
        """Detect content type from content"""
        if content.strip().startswith("{") and content.strip().endswith("}"):
            return "json"
        elif "<!DOCTYPE html>" in content.lower() or "<html" in content.lower():
            return "html"
        elif content.startswith("---") and "\n---\n" in content:
            return "markdown"
        return "text"

    async def _extract_structured_data(self, content: str) -> Dict[str, Any]:
        """Extract structured data from content"""
        structured_data = {}

        # Try to extract JSON-LD
        json_ld_matches = re.findall(
            r'<script type="application/ld\+json">(.*?)</script>', content, re.DOTALL
        )
        for match in json_ld_matches:
            try:
                data = json.loads(match)
                structured_data["json_ld"] = data
            except json.JSONDecodeError:
                pass

        # Try to extract Open Graph tags
        og_tags = re.findall(r'<meta property="og:(.*?)" content="(.*?)"', content)
        if og_tags:
            structured_data["open_graph"] = {tag: content for tag, content in og_tags}

        return structured_data

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content"""
        metadata = {}

        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", content)
        if title_match:
            metadata["title"] = title_match.group(1)

        # Extract meta description
        desc_match = re.search(r'<meta name="description" content="(.*?)"', content)
        if desc_match:
            metadata["description"] = desc_match.group(1)

        # Extract meta keywords
        keywords_match = re.search(r'<meta name="keywords" content="(.*?)"', content)
        if keywords_match:
            metadata["keywords"] = keywords_match.group(1).split(",")

        return metadata
