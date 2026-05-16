#!/bin/bash
# Configure CAN interfaces for Piper dual-arm setup.
# Run:  sudo bash scripts/setup_can.sh

set -e

BITRATE=${1:-1000000}

echo "=== Piper CAN Setup ==="

# can0 (follower)
if ip link show can0 &>/dev/null; then
    sudo ip link set can0 down 2>/dev/null || true
    sudo ip link set can0 up type can bitrate $BITRATE
    echo "can0 UP @ $BITRATE bps"
else
    echo "can0 not found — check hardware"
fi

# can1 (leader)
if ip link show can1 &>/dev/null; then
    sudo ip link set can1 down 2>/dev/null || true
    sudo ip link set can1 up type can bitrate $BITRATE
    echo "can1 UP @ $BITRATE bps"
else
    echo "can1 not found — check hardware (leader arm may not be connected)"
fi

echo "Done. Check: ip link show can0 can1"
