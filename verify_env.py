import sys

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
        return True
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        return False

print(f"\nPython: {sys.version}\n")

all_ok = True

all_ok &= check("SimpleITK import", lambda: __import__("SimpleITK"))
all_ok &= check("Elastix filter available", lambda: __import__("SimpleITK").ElastixImageFilter())
all_ok &= check("PCAMetric2 in groupwise map", lambda: (
    lambda sitk: sitk.GetDefaultParameterMap("groupwise").get("Metric") or
    (_ for _ in ()).throw(Exception("no Metric key"))
)(__import__("SimpleITK")))

import SimpleITK as sitk
def _check_pcametric2():
    pm = sitk.GetDefaultParameterMap("groupwise")
    pm["Metric"] = ["PCAMetric2"]
    pm["FixedImageDimension"] = ["4"]
    pm["MovingImageDimension"] = ["4"]
    f = sitk.ElastixImageFilter()
    f.SetParameterMap(pm)
all_ok &= check("PCAMetric2 accepted by Elastix", _check_pcametric2)

all_ok &= check("ANTsPy import", lambda: __import__("ants"))
all_ok &= check("bm3d import", lambda: __import__("bm3d"))
all_ok &= check("nibabel import", lambda: __import__("nibabel"))
all_ok &= check("scikit-image import", lambda: __import__("skimage"))

print()
if all_ok:
    print("All checks passed — environment is ready.")
else:
    print("Some checks failed — see above.")
print()
