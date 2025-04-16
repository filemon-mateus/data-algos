FROM alpine:3.21.3

# default mirror
ARG TL_MIRROR="https://mirror.ctan.org/systems/texlive/tlnet"

# install texlive
RUN apk add --no-cache perl wget && \
  mkdir /tmp/texlive && cd /tmp/texlive && \
  wget "$TL_MIRROR/install-tl-unx.tar.gz" && \
  tar xzvf ./install-tl-unx.tar.gz && \
  ( \
    echo "selected_scheme scheme-infraonly" && \
    echo "tlpdbopt_install_docfiles 0" && \
    echo "tlpdbopt_install_srcfiles 0" && \
    echo "TEXDIR /usr/local/texlive/" \
  ) > "/tmp/texlive.profile" && \
  "./install-tl-"*"/install-tl" -profile /tmp/texlive.profile

# add texlive binaries to path
ENV PATH="$PATH:/usr/local/texlive/bin/x86_64-linuxmusl"

# install latex packages
RUN tlmgr install \
  amsfonts \
  amsmath \
  caption \
  enumitem \
  epstopdf-pkg \
  etoolbox \
  fancyhdr \
  float \
  fontspec \
  geometry \
  hyperref \
  kpfonts-otf \
  lastpage \
  latex-bin \
  latexmk \
  listings \
  lualatex-math \
  pgfplots \
  realscripts \
  siunitx \
  tools \
  unicode-math

# clean up
RUN rm -vf /usr/local/texlive/install-tl && \
  rm -vf /usr/local/texlive/install-tl.log && \
  rm -vrf /tmp/*
