#!/bin/sh

MKDIR="/usr/bin/mkdir"
TAR="/usr/bin/tar"
LN="/usr/bin/ln"
BIN_DIR="$HOME/bin"
FF_DIR="$BIN_DIR/firefox"

FF_LINK="https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64-aarch64&lang=en-US"

if [[ ! -d "$FF_DIR" ]]; then
    # download firefox tarball and grab filename
    echo "Downloading firefox..."
    ff_file_name=$(wget -nv -t 20 --content-disposition "$FF_LINK"  2>&1 | cut -d\" -f2)

    # make HOME/bin directory if it doesn't exist
    $MKDIR -p $BIN_DIR

    # extract tarball into HOME/bin directory
    echo "Extracting firefox to $BIN_DIR..."
    $TAR -xf $ff_file_name --directory $BIN_DIR

    # linking firefox binary to ff
    echo "Linking $FF_DIR/firefox to ff..."
    $LN -s "$FF_DIR/firefox" "$BIN_DIR/ff"

    # add $BIN_DIR/ff to PATH
    echo "Adding $BIN_DIR/ff to PATH..."
    export PATH=$PATH:$BIN_DIR

    echo ""
    echo "To permanently add '$BIN_DIR' to your PATH, add 'export PATH=\$PATH:$BIN_DIR' to $HOME/.bashrc"
else
    echo "Firefox directory exists ($FF_DIR), skipping download..."
fi

