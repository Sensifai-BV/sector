/* RP2040 memory map.
 *
 * The first 256 bytes of flash hold the second-stage bootloader, which the
 * boot ROM checksums and copies to SRAM before anything else runs. The image
 * must therefore start at 0x10000100, not at the start of flash.
 *
 * FLASH length is the Pico's 2 MiB module. A board with a different module
 * needs this changed; the firmware also reads the real capacity at runtime, so
 * a mismatch is reported rather than silently truncating the volume.
 */
MEMORY {
    BOOT2 : ORIGIN = 0x10000000, LENGTH = 0x100
    FLASH : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    RAM   : ORIGIN = 0x20000000, LENGTH = 264K
}

SECTIONS {
    /* Placed by rp2040-boot2; must be the first thing in the image. */
    .boot2 ORIGIN(BOOT2) : {
        KEEP(*(.boot2));
    } > BOOT2
} INSERT BEFORE .text;
