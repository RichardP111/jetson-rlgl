#!/bin/bash

# ==============================================================================
# Red Light / Green Light - Command Center
#
#
# Author:       Richard Pu
# Last Updated: April 2026
# ==============================================================================

HEADLESS_FILE="/etc/X11/xorg.conf.d/99-headless.conf"

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🔴 RED LIGHT  /  GREEN LIGHT 🟢                          ║"
echo "║   Environment & Hardware Launcher                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "--- DISPLAY SETUP ---"
echo "  1) Enable School Mode (Physical DP Monitor)"
echo "  2) Enable Home Mode (GPU-Accelerated Headless)"
echo ""
echo "--- GAME ENGINE ---"
echo "  3) START RED LIGHT GREEN LIGHT"
echo ""
echo "--- HARDWARE TESTS ---"
echo "  4) Test Camera (test_camera.py)"
echo "  5) Test Servo (test_servo.py)"
echo "  6) Test Laser (test_laser.py)"
echo ""
echo "  7) Exit"
echo ""
read -p "Select an option [1-7]: " CHOICE

case $CHOICE in
    1)
        echo -e "\n=> [School Mode] Removing headless configuration..."
        sudo rm -f $HEADLESS_FILE
        echo "=> Restarting display manager..."
        sudo systemctl restart display-manager
        echo "Done! The physical monitor is now active on :0"
        ;;
    2)
        echo -e "\n=> [Home Mode] Forcing NVIDIA GPU to run headless..."
        sudo mkdir -p /etc/X11/xorg.conf.d/
        sudo bash -c 'cat > '$HEADLESS_FILE' <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen "Screen0"
EndSection

Section "Device"
    Identifier "Tegra0"
    Driver "nvidia"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "NoLogo" "true"
EndSection

Section "Screen"
    Identifier "Screen0"
    Device "Tegra0"
    SubSection "Display"
        Depth 24
        Virtual 1920 1080
    EndSubSection
EndSection
EOF'
        echo "=> Restarting display manager and NoMachine..."
        sudo systemctl restart display-manager
        sleep 3
        sudo /etc/NX/nxserver --restart
        echo "Done! You can now connect via NoMachine. The GPU is active!"
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
        sudo docker restart squid-game-live
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
        echo -e "\n--- Running Camera Test ---"
        sudo systemctl restart nvargus-daemon
        sleep 2
        export DISPLAY=:0
        xhost + 
        
        sudo docker restart squid-game-live
        sudo docker exec -it squid-game-live bash -c "
            export DISPLAY=:0 &&
            export XAUTHORITY=/root/.Xauthority &&
            cd /workspace &&
            python3 hardware_tests/test_camera.py
        "
        ;;
    5)
        echo -e "\n--- Running Servo Test ---"
        sudo docker restart squid-game-live
        sudo docker exec -it squid-game-live bash -c "
            export DISPLAY=:0 &&
            export XAUTHORITY=/root/.Xauthority &&
            cd /workspace &&
            python3 hardware_tests/test_servo.py
        "
        ;;
    6)
        echo -e "\n--- Running Laser Test ---"
        sudo docker restart squid-game-live
        sudo docker exec -it squid-game-live bash -c "
            cd /workspace &&
            python3 hardware_tests/test_laser.py
        "
        ;;
    7)
        echo -e "\nExiting..."
        exit 0
        ;;
    *)
        echo -e "\nInvalid choice. Exiting."
        exit 1
        ;;
esac