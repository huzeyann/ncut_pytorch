# %%
from ncut_pytorch_git.ncut_pytorch.predictor.dino.transform import get_input_transform
tsf = get_input_transform(resize=(2048, 2048))
# %%

import time
def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return result
    return wrapper

dinov3_urls = {
    "dinov3_vits16": "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vits16plus": "https://dinov3.llamameta.net/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vitb16": "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vitl16": "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vith16plus": "https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vit7b16": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    # SAT-493M variants
    "dinov3_vitl16_sat493m": "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vit7b16_sat493m": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_sat493m-a6675841.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    # ConvNeXT variants
    "dinov3_convnext_tiny": "https://dinov3.llamameta.net/dinov3_convnext_tiny/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_convnext_small": "https://dinov3.llamameta.net/dinov3_convnext_small/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_convnext_base": "https://dinov3.llamameta.net/dinov3_convnext_base/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_convnext_large": "https://dinov3.llamameta.net/dinov3_convnext_large/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    # Adapter heads
    "dinov3_vit7b16_imagenet1k_linear_head": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vit7b16_coco_detr_head": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_coco_detr_head-b0235ff7.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vit7b16_ade20k_m2f_head": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vit7b16_synthmix_dpt_head": "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
    "dinov3_vitl16_dinotxt_vision_head_and_text_encoder": "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZnVqa3l4cXl1emd2enoxbnN5aHVmcGg1IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0NTQ3MzN9fX1dfQ__&Signature=kQ219YVhXExdwrmWMrkeciFz1%7Ep8IByF7h4N2HISgR7JA4PJclm2v7VNs1CO6him4O1vTeFtEcDOStsoxKa0cZMoLKZZ2sszILXyW6fw-4KCGfbackP3jXhsGjGbJQguxexDD7nzaOQgJHFKBXC0ozn9GBrd9r5d4y8No-qKiQ2kJZ6i7nrHXNFw1biPdBv-yjNyjfr-YQlHErFW3Miyhmn8wGymydwjOWsOewMVakKcZ4R-yqm%7E3RRKB%7EnnVmo6AjChmB9v8OoaD2Ac%7Ehfkml2K8kJij6pClrPxKHkrzGxg1P0ThQsPQLoZdqCyJnxtNTVlAoskwiY3S8MSoMuVuQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082585944013465",
}


import torch
from torch import nn
class Dinov3Backbone(nn.Module):
    def __init__(self, config="dinov3_vith16plus"):
        super().__init__()
        dinov3 = torch.hub.load("facebookresearch/dinov3", config, weights=dinov3_urls[config])
        self.model = dinov3

    @timeit_decorator
    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        return self.model.get_intermediate_layers(x, reshape=True)[0]
# %%
from torch import Tensor

def _get_attn_masking_fn(mask_ratio: float):
    
    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        
        num_keys = int(N * mask_ratio)
        mask_indices = torch.randperm(N-5)[:num_keys] + 5
        mask_indices = torch.cat((torch.arange(0, 5), mask_indices)) # add the CLS token
        k = k[:, :, mask_indices, :]
        v = v[:, :, mask_indices, :]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
    
    return compute_attention
# %%

def apply_masked_attention_to_model(model, mask_ratio=0.1):
    """
    Replace the compute_attention method in all attention layers with the masked version
    """
    new_compute_attention = _get_attn_masking_fn(mask_ratio)
    
    # Apply to all transformer blocks
    for block in model.model.blocks:
        if hasattr(block, 'attn'):
            # Replace the compute_attention method
            import types
            block.attn.compute_attention = types.MethodType(new_compute_attention, block.attn)
    
    return model

# %%
# First create the model and apply masked attention
backbone_model = Dinov3Backbone(config="dinov3_vith16plus")
# backbone_model = apply_masked_attention_to_model(backbone_model, mask_ratio=0.5)

from ncut_pytorch_git.ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor
predictor = NcutVisionPredictor(model=backbone_model, transform=tsf, batch_size=1)
# from ncut_pytorch_git.ncut_pytorch.predictor import NcutDinoPredictorSR
# predictor = NcutDinoPredictorSR(dtype=torch.float16)
# %%
predictor = predictor.to('cuda')
from PIL import Image


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]
start_time = time.time()
predictor.set_images(images)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# %%
predictor.predictor.refresh_color_palette()
# %%
image = predictor.summary(draw_border=True)
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(image)
ax.axis('off')
ax.set_title('Dinov3 vitl16')
plt.show()
# %%