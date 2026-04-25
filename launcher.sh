#!/bin/bash

# ==============================================================================
# Red Light / Green Light - Command Center
# ==============================================================================

DUMMY_FILE="/etc/X11/xorg.conf.d/99-dummy.conf"

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🔴 RED LIGHT  /  GREEN LIGHT 🟢                          ║"
echo "║   Environment & Hardware Launcher                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "--- DISPLAY SETUP ---"
echo "  1) Enable School Mode (Physical DP Monitor)"
echo "  2) Enable Home Mode (Headless / NoMachine Remote)"
echo ""
echo "--- GAME ENGINE ---"
echo "  3) START RED LIGHT GREEN LIGHT"
echo "  4) Exit"
echo ""
read -p "Select an option [1-4]: " CHOICE

case $CHOICE in
    1)
        echo -e "\n=> [School Mode] Removing virtual display driver..."
        sudo rm -f $DUMMY_FILE
        echo "=> Restarting display manager..."
        sudo systemctl restart display-manager
        echo "Done! The physical monitor is now active."
        ;;
    2)
        echo -e "\n=> [Home Mode] Injecting 1080p virtual display driver..."
        sudo mkdir -p /etc/X11/xorg.conf.d/
        sudo bash -c 'cat > '$DUMMY_FILE' <<EOF
Section "Device"
    Identifier "DummyDevice"
    Driver "dummy"
    VideoRam 256000
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-120.0
EndSection

Section "Screen"
    Identifier "DummyScreen"
    Device "DummyDevice"
    Monitor "DummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
EOF'
        echo "=> Restarting display manager..."
        sudo systemctl restart display-manager
        echo "Done! You can now connect via NoMachine."
        ;;
    3)
        echo -e "\n--- [1/3] Clearing Hardware Locks ---"
        pulseaudio -k 2>/dev/null || true
        sudo systemctl restart nvargus-daemon
        sleep 2

        echo -e "\n--- [2/3] Configuring Display Permissions ---"
        export DISPLAY=:0
        xhost + 

        echo -e "\n--- [3/3] Booting Game Engine ---"
        sudo docker start squid-game-live
        sudo docker exec -it squid-game-live bash -c "
            export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:\$LD_LIBRARY_PATH &&
            export SDL_AUDIODRIVER=alsa &&
            export AUDIODEV=plughw:2,0 &&
            export DISPLAY=:0 &&
            export XAUTHORITY=/root/.Xauthority &&
            export XDG_RUNTIME_DIR=/tmp/runtime-root &&
            mkdir -p \$XDG_RUNTIME_DIR &&
            chmod 700 \$XDG_RUNTIME_DIR &&
            cd /workspace &&
            python3 main.py
        "
        echo -e "\nGame instance closed. Hardware locks released."
        ;;
    4)
        echo -e "\nExiting..."
        exit 0
        ;;
    *)
        echo -e "\nInvalid choice. Exiting."
        exit 1
        ;;
esac