# README media provenance

These assets illustrate **Vision-Based 6-DoF Grasp Pose Estimation for Robot Cloth Unfolding**, IEEE Transactions on Cybernetics (2026), DOI [10.1109/TCYB.2026.3732543](https://doi.org/10.1109/TCYB.2026.3732543).

| Asset | Source and preparation |
| --- | --- |
| `architecture.png` | Rendered at 1800 pixels wide from the authors' figure source `images/intro_revision_6DoF_compressed.pdf`, included in the author-prepared arXiv source package. This is Figure 1, not a crop of the publisher PDF. |
| `grasp-results.jpg` | Rendered at 1800 pixels wide from the authors' `images/detections_compressed.pdf` (Figure 4), then JPEG-encoded at quality 90. |
| `supplementary-video.jpg` | Copied without alteration from the ViCoS website's CeDiRNet-6DoF research-page asset `images/final-montage.jpg`; the site's existing poster for the supplementary video. |

The original author-source figures and website poster were visually checked; no publisher watermark was used. Image reuse remains subject to the authors' and applicable publication rights; the repository's code license should not be interpreted as a separate blanket grant for paper figures or video.

## Supplementary video

The public URL is taken directly from the ViCoS website's publication and research-page source:

https://data.vicos.si/skokec/rtfm/CeDiRNet-6DoF/cedirnet_demo_h264_1080p_submission.mp4

Verified with an HTTP range request: `206 Partial Content`, `Content-Type: video/mp4`, valid MP4 `ftyp` header, total length 7,169,515 bytes. The README uses a clickable local poster rather than an HTML video element, which GitHub may strip. No GitHub-uploaded video attachment URL was found or invented; the MP4 is hosted externally and is not stored in this repository.
