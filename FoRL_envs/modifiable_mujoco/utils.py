import contextlib
import tempfile
import os
import xml.etree.ElementTree as ET


class FoRLXMLModifierMixin:
    """Mixin with XML modification methods."""

    @contextlib.contextmanager
    def modify_xml(self, original_path):
        """Context manager allowing XML asset modification."""

        try:
            fd, path = tempfile.mkstemp(suffix=".xml")
            tree = ET.parse(original_path)
            yield tree
            os.close(fd)
            tree.write(path)

            self.fullpath = path

        finally:
            if os.path.exists(path):
                os.remove(path)
