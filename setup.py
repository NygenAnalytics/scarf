import glob
import os

from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """Copy bundled binaries from bin/ into the install scripts directory."""

    def run(self):
        install.run(self)
        if not os.path.isdir(self.install_scripts):
            os.makedirs(self.install_scripts)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        source_files = glob.glob(os.path.join(source_dir, "bin") + "/*")
        for source in source_files:
            target = os.path.join(self.install_scripts, os.path.basename(source))
            if os.path.isfile(target):
                os.remove(target)
            with open(source, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
            os.chmod(target, 0o755)


setup(cmdclass={"install": PostInstallCommand})
