def test_imports():
    import tensorflow as tf
    import cv2
    assert tf.__version__
    assert cv2.__version__