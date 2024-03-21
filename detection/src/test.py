# from docarray import DocList, DocVec
#
# from executor import DetectionExecutor
# from doc import ImageUrlDoc
#
# from PIL import Image
#
# ex = DetectionExecutor()
#
# l = DocVec[ImageUrlDoc]([ImageUrlDoc(url="https://www.purina.co.uk/sites/default/files/2023-03/Hero%20Pedigree%20Cats.jpg")])
#
# out = ex.detect(docs=l)
# for i, doc in enumerate(out):
#     Image.fromarray(doc.crop).save(f"{i}.png")
# print(1)
