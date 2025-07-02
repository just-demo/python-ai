import os
import sys

from mcp.server.fastmcp import FastMCP

sandbox_dir = os.path.abspath(sys.argv[1])

mcp = FastMCP("custom_server")


@mcp.tool()
async def write_file(file: str, content: str):
    """Write the given content to a file. If the file already exists, the content will be appended.

    Args:
        file: file name to write to
        content: file content to write
    """
    file_path = os.path.join(sandbox_dir, file)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
    # Need to return something, otherwise the agent would report an error
    return "ok"


if __name__ == "__main__":
    mcp.run(transport="stdio")
