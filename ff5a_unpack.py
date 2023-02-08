#! python3
# coding: utf-8

import os, os.path

def report(*args):
    r = ' '.join(args)
    print(r)
    return r

alignup   = lambda v, a: ((v - 1) // a + 1) * a
aligndown = lambda v, a: (v // a) * a

def readval_le(raw, offset, size, signed):
    neg = False
    v = 0
    endpos = offset + size - 1
    for i in range(endpos, offset - 1, -1):
        b = raw[i]
        if signed and i == endpos and b > 0x7f:
            neg = True
            b &= 0x7f
        #else:
        #    b &= 0xff
        v <<= 8
        v += b
    return v - (1 << (size*8 - 1)) if neg else v

def writeval_le(val, dst, offset, size):
    if val < 0:
        val += (1 << (size*8))
    for i in range(offset, offset + size):
        dst[i] = (val & 0xff)
        val >>= 8

INF = float('inf')

class c_mark:

    def __init__(self, raw, offset):
        self._raw = raw
        self._mod = None
        self.offset = offset
        self.parent = None
        self._par_offset = 0

    @property
    def raw(self):
        if self.parent:
            return self.parent.raw
        return self._mod if self._mod else self._raw

    @property
    def mod(self):
        if self.parent:
            return self.parent.mod
        if not self._mod:
            self._mod = bytearray(self._raw)
        return self._mod

    @property
    def par_offset(self):
        po = self._par_offset
        if self.parent:
            po += self.parent.par_offset
        return po

    def shift(self, offs):
        self._par_offset += offs

    def extendto(self, cnt):
        extlen = self.offset + cnt - len(self.raw)
        if extlen > 0:
            self.mod.extend(bytes(extlen))

    def readval(self, pos, cnt, signed):
        return readval_le(self.raw, self.offset + pos, cnt, signed)

    def writeval(self, val, pos, cnt):
        self.extendto(pos + cnt)
        writeval_le(val, self.mod, self.offset + pos, cnt)

    def fill(self, val, pos, cnt):
        for i in range(pos, pos + cnt):
            self.mod[i] = val

    def findval(self, val, pos, cnt, signed):
        st = pos
        ed = len(self.raw) - cnt + 1 - st
        for i in range(st, ed, cnt):
            s = self.readval(i, cnt, signed)
            if s == val:
                return i
        else:
            return -1

    I8  = lambda self, pos: self.readval(pos, 1, True)
    U8  = lambda self, pos: self.readval(pos, 1, False)
    I16 = lambda self, pos: self.readval(pos, 2, True)
    U16 = lambda self, pos: self.readval(pos, 2, False)
    I32 = lambda self, pos: self.readval(pos, 4, True)
    U32 = lambda self, pos: self.readval(pos, 4, False)
    I64 = lambda self, pos: self.readval(pos, 8, True)
    U64 = lambda self, pos: self.readval(pos, 8, False)

    W8  = lambda self, val, pos: self.writeval(val, pos, 1)
    W16 = lambda self, val, pos: self.writeval(val, pos, 2)
    W32 = lambda self, val, pos: self.writeval(val, pos, 4)
    W64 = lambda self, val, pos: self.writeval(val, pos, 8)

    FI8 = lambda self, val, pos: self.findval(val, pos, 1, True)
    FU8 = lambda self, val, pos: self.findval(val, pos, 1, False)
    FI16 = lambda self, val, pos: self.findval(val, pos, 2, True)
    FU16 = lambda self, val, pos: self.findval(val, pos, 2, False)
    FI32 = lambda self, val, pos: self.findval(val, pos, 4, True)
    FU32 = lambda self, val, pos: self.findval(val, pos, 4, False)
    FI64 = lambda self, val, pos: self.findval(val, pos, 8, True)
    FU64 = lambda self, val, pos: self.findval(val, pos, 8, False)

    def BYTES(self, pos, cnt):
        st = self.offset + pos
        if cnt is None:
            ed = None
        else:
            ed = st + cnt
            self.extendto(pos + cnt)
        return self.raw[st: ed]

    def STR(self, pos, cnt, codec = 'utf8'):
        return self.BYTES(pos, cnt).split(b'\0')[0].decode(codec)

    def BYTESN(self, pos):
        st = self.offset + pos
        rl = len(self.raw)
        ed = rl
        for i in range(st, rl):
            if self.raw[i] == 0:
                ed = i
                break
        return self.raw[st:ed], ed - st

    def STRN(self, pos, codec = 'utf8'):
        b, n = self.BYTESN(pos)
        return b.decode(codec), n

    def FBYTES(self, dst, pos, stp = 1):
        cnt = len(dst)
        st = self.offset + pos
        ed = len(self.raw) - cnt + 1
        for i in range(st, ed, stp):
            for j in range(cnt):
                if self.raw[i+j] != dst[j]:
                    break
            else:
                return i - self.offset
        else:
            return -1

    def FSTR(self, dst, pos, stp = 1, codec = 'utf8'):
        return self.FBYTES(dst.encode(codec), pos, stp)

    def sub(self, pos, length = 0, cls = None):
        if not cls:
            cls = c_mark
        if length > 0:
            s = cls(None, 0)
            s._mod = bytearray(self.BYTES(pos, length))
            s._par_offset = self.par_offset + pos
        else:
            s = cls(None, self.offset + pos)
            s.parent = self
        return s

class c_ff5a_sect(c_mark):

    _SIGN_PAD = '\0\0\0\0'
    def _find_next_sign(self, sign, pos, align = 4):
        dpos = self.FSTR(self._SIGN_PAD + sign.upper(),
            aligndown(pos, align), align)
        if dpos < 0:
            return -1, -1
        else:
            return dpos, dpos + align

class c_ff5a_sect_tab(c_ff5a_sect):

    def parse(self):
        self._parse_head()

    def _parse_head(self):
        return NotImplemented

    def _parse_index(self, pos):
        mk = self.sub(pos)
        nxt_pos = mk.U32(0x0)
        self.mk_index = mk
        self.len_index = nxt_pos - pos
        self.cnt_index = self.len_index // 4

    def _get_pos(self, idx):
        return self.mk_index.U32(idx * 4)

    def get_item(self, idx):
        if not 0 <= idx < self.cnt_index:
            return -1, -1
        dpos = self._get_pos(idx)
        if idx < self.cnt_index - 1:
            npos = self._get_pos(idx+1)
        else:
            npos = self.pos_last
        return dpos, npos - dpos

class c_ff5a_sect_text(c_ff5a_sect_tab):

    def _parse_head(self):
        self.pos_last = self.U32(0xc)
        self._parse_index(0x10)

    def get_text_raw(self, idx, detail = False):
        dpos, dlen = self.get_item(idx)
        rtxt = self.BYTES(dpos, dlen)
        if detail:
            return rtxt, dpos, dlen
        else:
            return rtxt

class c_ff5a_sect_font(c_ff5a_sect_tab):

    def _parse_head(self):
        cnt_tab = self.U16(0xa)
        font_width = self.U8(0x8)
        font_pad = self.U8(0x9)
        self._parse_index(0x10c)
        assert cnt_tab == self.cnt_index
        sz_font = font_width * font_width * 2 // 8 + font_pad
        self.pos_last = self._get_pos(cnt_tab - 1) + sz_font
        self.font_width = font_width
        self.font_pad = font_pad
        self.font_size = sz_font

class c_ff5a_sect_rom(c_ff5a_sect):

    def __init__(self, raw, offset):
        super().__init__(raw, offset)
        self._parse_tabs({
            'text': c_ff5a_sect_text,
            'font': c_ff5a_sect_font,
        })

    def _parse_tabs(self, tabs_info):
        tabs = {}
        for sign in tabs_info:
            tab = []
            pos = 0
            c_tab = tabs_info[sign]
            while True:
                dpos, pos = self._find_next_sign(sign, pos)
                if dpos <0:
                    break
                report('info', f'found tab {sign}[{len(tab)}] at 0x{dpos:x}')
                mk = self.sub(dpos, cls = c_tab)
                mk.parse()
                tab.append(mk)
            if tab:
                tabs[sign] = tab
        self.tabs = tabs

class c_ff5a_parser:

    def __init__(self, path):
        self.rom_path = path

    def load_rom(self):
        with open(self.rom_path, 'rb') as fd:
            raw = fd.read()
        self.rom = c_ff5a_sect_rom(raw, 0)

if __name__ == '__main__':
    from hexdump import hexdump
    #psr = c_ff5a_parser('ff5ajp.gba')
    psr = c_ff5a_parser('ff5acn.gba')
    psr.load_rom()
    txtjp = psr.rom.tabs['text'][0]
    txtcn = psr.rom.tabs['text'][1]
    def _show_long_txt(txt, th = 8):
        for i in range(txt.cnt_index):
            s, p, l = txt.get_text_raw(i, True)
            if l > th:
                print(f'{i}:+0x{p:x}(0x{l:x})')
                hexdump(s)
                yield
    slt = _show_long_txt(txtjp, 0x20)
    def _cmp_long_txt(th):
        tcnt = min(txtjp.cnt_index, txtcn.cnt_index)
        print(f'jp:{txtjp.cnt_index}, cn:{txtcn.cnt_index}')
        for i in range(tcnt):
            s1, p1, l1 = txtjp.get_text_raw(i, True)
            s2, p2, l2 = txtcn.get_text_raw(i, True)
            if l1 == l2 > th:
                print(f'jp {i}:+0x{p1:x}(0x{l1:x})')
                hexdump(s1)
                print(f'cn {i}:+0x{p2:x}(0x{l2:x})')
                hexdump(s2)
                yield
    clt = _cmp_long_txt(0x20)
    next(clt)
