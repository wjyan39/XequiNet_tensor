from setuptools import setup
setup(
    name = "XequiNet",
    version = "0.3.0",
    packages = ["xequinet"],
    entry_points={
        'console_scripts': [
            "xeqtrain = xequinet.run.train:main",
            "xeqjit = xequinet.run.jit_script:main",
            "xeqinfer = xequinet.run.inference:main",
            "xeqtest = xequinet.run.test:main",
            "xeqopt = xequinet.run.geometry:main",
        ]
    }
)