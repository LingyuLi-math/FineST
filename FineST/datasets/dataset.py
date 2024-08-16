from copy import copy
from ._util_dataset import AMetadata


_NPC = AMetadata(
    name="NPC",
    doc_header="Nasopharyngeal carcinoma (NPC) dataset from `Gong et al <https://doi.org/10.1038/s41467-023-37614-6>`__.",
    shape=(1331, 36601),
    url="https://figshare.com/ndownloader/files/48619396",
)

_BRCA = AMetadata(
    name="BRCA",
    doc_header="Breast cancer (BRCA) dataset from `Janesick et al <https://doi/10.1038/s41467-023-43458-x`__.",
    shape=(4992, 18085),
    url="https://ndownloader.figshare.com/files/40178041",
)

_CRC = AMetadata(
    name="CRC",
    doc_header="Adult colon rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(2649, 33538),
    url="https://ndownloader.figshare.com/files/40178029",
)

_A2 = AMetadata(
    name="A2",
    doc_header="Adult colon rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(2316, 33538),
    url="https://figshare.com/ndownloader/files/40178317",
)

_A3 = AMetadata(
    name="A3",
    doc_header="12-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1080, 33538),
    url="https://figshare.com/ndownloader/files/40178311",
)

_A4 = AMetadata(
    name="A4",
    doc_header="19-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1242, 33538),
    url="https://figshare.com/ndownloader/files/40178314",
)

_A6 = AMetadata(
    name="A6",
    doc_header="12-PCW TI rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(346, 33538),
    url="https://figshare.com/ndownloader/files/40178017",
)

_A7 = AMetadata(
    name="A7",
    doc_header="12-PCW TI rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(344, 33538),
    url="https://figshare.com/ndownloader/files/40178014",
)

_A8 = AMetadata(
    name="A8",
    doc_header="12-PCW colon rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(709, 33538),
    url="https://figshare.com/ndownloader/files/40178011",
)

_A9 = AMetadata(
    name="A9",
    doc_header="12-PCW colon rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(644, 33538),
    url="https://figshare.com/ndownloader/files/40178308",
)


for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "NPC", "BRCA",
    "CRC","A2","A3","A4","A6","A7","A8","A9"]
