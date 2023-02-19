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
        ed = len(self.raw) - cnt + 1 - self.offset
        for i in range(st, ed, cnt):
            s = self.readval(i, cnt, signed)
            if s == val:
                return i
        else:
            return -1

    def forval(self, cb, pos, cnt, signed):
        st = pos
        ed = len(self.raw) - cnt + 1 - self.offset
        ln = (ed - st + 1) // 2
        for i in range(st, ed, cnt):
            s = self.readval(i, cnt, signed)
            if cb(i, s, ln) == False:
                return False
        return True

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

    FORI8 = lambda self, cb, pos: self.forval(cb, pos, 1, True)
    FORU8 = lambda self, cb, pos: self.forval(cb, pos, 1, False)
    FORI16 = lambda self, cb, pos: self.forval(cb, pos, 2, True)
    FORU16 = lambda self, cb, pos: self.forval(cb, pos, 2, False)
    FORI32 = lambda self, cb, pos: self.forval(cb, pos, 4, True)
    FORU32 = lambda self, cb, pos: self.forval(cb, pos, 4, False)
    FORI64 = lambda self, cb, pos: self.forval(cb, pos, 8, True)
    FORU64 = lambda self, cb, pos: self.forval(cb, pos, 8, False)

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

    def concat(self, dst, pos = None):
        db = dst.BYTES(0, None)
        if pos is None:
            self.mod.extend(db)
        else:
            sb = self.BYTES(0, pos)
            self.extendto(pos + len(db))
            self.mod[pos:] = db
        return self

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
        self.len_head = pos
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

    def get_raw(self, idx, detail = False):
        dpos, dlen = self.get_item(idx)
        rtxt = self.BYTES(dpos, dlen)
        if detail:
            return rtxt, dpos, dlen
        else:
            return rtxt

    def get_offset(self, idx):
        pos, ln = self.get_item(idx)
        return self.offset + pos

    def find_item_by_offset(self, offset):
        pos = offset - self.offset
        if pos < 0:
            return -1
        for i in range(self.cnt_index):
            dpos, dlen = self.get_item(i)
            if dpos <= pos < dpos + dlen:
                return i, dpos + self.offset
            elif pos < dpos:
                return -1

    def repack_with(self, rplc):
        rmk = self.sub(0, self.len_head, type(self))
        offs = self.len_head
        rmk_index = rmk.sub(offs)
        offs += self.cnt_index * 4
        for i in range(self.cnt_index):
            if i in rplc:
                dat = rplc[i]
            else:
                dat = self.get_raw(i)
            rmk_index.W32(offs, i * 4)
            for j, c in enumerate(dat):
                rmk.W8(c, offs + j)
            offs += len(dat)
        rmk.pos_last = offs
        rmk._repack_head()
        report('info', f'repack 0x{rmk.pos_last:x} bytes')
        rmk.parse()
        return rmk

    def _repack_head(self):
        return NotImplemented

class c_ff5a_sect_text(c_ff5a_sect_tab):

    def _parse_head(self):
        self.pos_last = self.U32(0xc)
        self._parse_index(0x10)
        report('info', f'found {self.cnt_index} texts')

    def _repack_head(self):
        self.W32(self.pos_last, 0xc)

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
        self.font_height = font_width
        self.font_pad = font_pad
        self.font_size = sz_font
        report('info', f'found 0x{self.cnt_index:x} chars')

    def _repack_head(self):
        # nothing to change
        pass

class c_ff5a_sect_rom(c_ff5a_sect):

    BASE_ADDR = 0x8000000

    def parse(self):
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
        self._parse_entries()

    def _parse_entries(self):
        entries = {}
        ent_r = {}
        min_tab = INF
        for k, tab in self.tabs.items():
            ent = []
            entries[k] = ent
            for i, mk in enumerate(tab):
                pos = mk.offset
                ent_r[pos + self.BASE_ADDR] = (k, i)
                ent.append([])
                if pos < min_tab:
                    min_tab = pos
        def find(pos, val, cnt):
            if pos >= min_tab:
                return False
            if val in ent_r:
                sign, idx = ent_r[val]
                report('info', f'found entry {sign}[{idx}] at 0x{pos:x}')
                entries[sign][idx].append(pos)
        self.FORU32(find, 0)
        self.entries = entries

    def _repack_entries(self, rmk, soffs, doffs):
        shft = doffs - soffs
        if shft == 0:
            return
        for k, tab in self.tabs.items():
            for i, mk in enumerate(tab):
                moffs = mk.offset
                if mk.offset < soffs:
                    continue
                dmoffs = moffs + shft
                report('info', f'shift tab from 0x{moffs:x} to 0x{dmoffs}')
                for ent in self.entries[k][i]:
                    assert rmk.U32(ent) == moffs + self.BASE_ADDR
                    rmk.W32(dmoffs + self.BASE_ADDR, ent)

    def repack_with(self, smk, dmk,
            align = 0x10, padding = 0xff):
        offs = smk.offset
        slen = alignup(smk.pos_last, align)
        dlen = alignup(dmk.pos_last, align)
        padlen = dlen - dmk.pos_last
        rmk = self.sub(0, offs, type(self))
        rmk_tail = self.sub(offs + slen)
        if padlen > 0:
            rmk_pad = c_mark(bytearray([padding] * padlen), 0)
            rmk.concat(dmk).concat(rmk_pad).concat(rmk_tail)
        else:
            rmk.concat(dmk).concat(rmk_tail)
        self._repack_entries(rmk, offs + slen, offs + dlen)
        rmk.parse()
        return rmk

class c_text_drawer:

    def __init__(self, font_tab):
        self.font = font_tab

    _pil_img = None
    @property
    def pil_img(self):
        if self._pil_img:
            return self._pil_img
        try:
            from PIL import Image, ImageDraw
        except:
            print('''
please install Pillow with
pip3 install pillow
or
pip install pillow
or
py -3 -m pip install pillow
or
python -m pip install pillow''')
            raise
        c_text_drawer._pil_img = Image
        c_text_drawer.pil_drw = ImageDraw
        return Image

##    PAL = [
##        (255, 255, 255), (230, 230, 230), (200, 200, 200), (0, 0, 0),
##        (240, 0, 240), (240, 240, 0), (0, 240, 240),
##        (0, 240, 0), (0, 0, 240), (240, 0, 0),
##        (120, 0, 120), (120, 120, 0), (0, 120, 120),
##        (0, 120, 0), (0, 0, 120), (120, 0, 0),
##    ]
    PAL = [(255, 255, 255), (100, 100, 100), (200, 200, 200), (0, 0, 0)]

    def draw_block(self, text, cols, pad_col = 3, pad_row = 5, bits = 2):
        tlen = len(text)
        pal = self.PAL
        font = self.font
        fmx = font.cnt_index
        fw = font.font_width
        fh = font.font_height
        fp = font.font_pad
        dpb = 8 // bits
        bmsk = (1 << bits) - 1
        assert fw % 2 == 0
        for i in range(0, tlen, cols):
            ed_row = min(i + cols, tlen)
            rlen = ed_row - i
            row = text[i: ed_row]
            for y in range(fh):
                rline = []
                for ch_ridx in range(cols):
                    if ch_ridx < rlen:
                        ch = row[ch_ridx]
                        if ch < fmx:
                            ch_pos, ch_sz = font.get_item(ch)
                            ch_pos += fp
                            for x in range(0, fw, dpb):
                                p_pos = ch_pos + (y * fw + x) // dpb
                                px = font.U8(p_pos)
                                for _ in range(dpb):
                                    rline.append(pal[px & bmsk])
                                    px >>= bits
                        else:
                            for x in range(fw):
                                rline.append(pal[2])
                    else:
                        for x in range(fw):
                            rline.append(pal[0])
                    if ch_ridx < cols - 1:
                        for x in range(pad_col):
                            rline.append(pal[0])
                yield rline, True
            if i + cols < tlen:
                for y in range(pad_row):
                    rline = []
                    for x in range((fw + pad_col) * cols - pad_col):
                        rline.append(pal[0])
                    yield rline, True
        while True:
            rline = []
            for x in range((fw + pad_col) * cols - pad_col):
                rline.append(pal[0])
            yield rline, False

    @staticmethod
    def draw_padding(width, height):
        clr_blank = c_text_drawer.PAL[0]
        for y in range(height):
            rline = []
            for x in range(width):
                rline.append(clr_blank)
            yield rline, True
        while True:
            rline = []
            for x in range(width):
                rline.append(clr_blank)
            yield rline, False

    def draw_comment(self, width, height, txt):
        pal = self.PAL
        im = self.pil_img.new('RGB', (width, height), pal[0])
        dr = self.pil_drw.Draw(im)
        dr.text((0, 0), txt, fill = pal[2])
        sq = im.getdata()
        w = im.width
        h = im.height
        for y in range(h):
            p = y * w
            rline = []
            for x in range(w):
                v = sq[p + x]
                rline.append(v)
            yield rline, True
        while True:
            rline = []
            for x in range(w):
                rline.append(pal[0])
            yield rline, False

    @staticmethod
    def draw_horiz(*blks, pad = 5):
        clr_blank = c_text_drawer.PAL[0]
        rwidth = 0
        while True:
            unfinished = False
            rline = []
            blen = len(blks)
            for i in range(blen):
                blk = blks[i]
                rl, uf = next(blk)
                if uf:
                    unfinished = True
                rline.extend(rl)
                if i < blen -1:
                    for x in range(pad):
                        rline.append(clr_blank)
            if not rwidth:
                rwidth = len(rline)
            if unfinished:
                yield rline, True
            else:
                break
        while True:
            rline = []
            for x in range(rwidth):
                rline.append(clr_blank)
            yield rline, False

    @staticmethod
    def draw_vert(*blks, pad = 10):
        clr_blank = c_text_drawer.PAL[0]
        blk_info = []
        for blk in blks:
            rl, uf = next(blk)
            if uf:
                blk_info.append((blk, rl, len(rl)))
        rwidth = max(p[2] for p in blk_info)
        blen = len(blk_info)
        for i in range(blen):
            blk, rl, rlen = blk_info[i]
            rl_pad = []
            for x in range(rwidth - rlen):
                rl_pad.append(clr_blank)
                rl.append(clr_blank)
            yield rl, True
            for rl, uf in blk:
                if not uf:
                    break
                if rl_pad:
                    rl.extend(rl_pad)
                yield rl, True
            if i < blen - 1:
                for y in range(pad):
                    rline = []
                    for x in range(rwidth):
                        rline.append(clr_blank)
                    yield rline, True
        while True:
            rline = []
            for x in range(rwidth):
                rline.append(clr_blank)
            yield rline, False

    def make_img(self, blk):
        dat = []
        bh = 0
        bw = 0
        for rl, uf in blk:
            if not uf:
                break
            if not bw:
                bw = len(rl)
            dat.extend(rl)
            bh += 1
        im = self.pil_img.new('RGB', (bw, bh))
        im.putdata(dat)
        return im

class c_ff5a_parser_text:

    def __init__(self, txt, fnt):
        self.text = txt
        self.font = fnt
        self.draw = c_text_drawer(fnt)

    def dec_text(self, tpos, tlen, tidx):
        r = []
        for i in range(tlen):
            pi = tpos + i
            c = self.text.U8(pi)
            r.append(c)
        return r

    def enc_text(self, txt):
        return txt

    def get_text(self, idx):
        tpos, tlen = self.text.get_item(idx)
        return self.dec_text(tpos, tlen, idx)

    def repack_text(self, rplc):
        return self.text.repack_with(
            {i: self.enc_text(t) for i, t in rplc.items()})

    def find_chars(self, chars, first = True, strict = False, ret_txt = False):
        hd = chars[0]
        slen = len(chars)
        r = []
        for idx in range(self.text.cnt_index):
            txt = self.get_text(idx)
            tlen = len(txt)
            if strict and not (
                tlen == slen or
                (tlen == slen + 1 and txt[-1] == 0)):
                continue
            for i in range(tlen - slen + 1):
                d = txt[i]
                if d != hd:
                    continue
                for j in range(1, slen):
                    sc = chars[j]
                    dc = txt[i + j]
                    if sc != dc:
                        break
                else:
                    if ret_txt:
                        r.append((idx, i, txt))
                    else:
                        r.append((idx, i))
                    if first:
                        return r
                    else:
                        break
        return r

    def draw_chars(self, chars, cols = None, pad_col = 3, pad_row = 5):
        if cols is None:
            cols = len(chars)
            if not cols:
                cols = 1
        return self.draw.draw_block(chars, cols, pad_col, pad_row)

    def draw_text(self, tidx, cols = None, pad_col = 3, pad_row = 5):
        txt = self.get_text(tidx)
        return self.draw_chars(txt, cols, pad_col, pad_row)

    def draw_comment(self, cmt, fw = 8, fh = 12):
        s = str(cmt)
        return self.draw.draw_comment(fw*len(s), fh, s)

    @staticmethod
    def is_ctrl(c):
        return (c & 0xf000) == 0xf000

    def dec_ctrl(self, c):
        if not self.is_ctrl(c):
            return 0
        return (c & 0xfff) - 0x4de

    def enc_ctrl(self, c):
        return (c + 0x4de) | 0xf000

class c_ff5a_parser_text_jp(c_ff5a_parser_text):

    def dec_text(self, tpos, tlen, tidx, mark_ctrl = True):
        fmx = self.font.cnt_index
        r = []
        i = 0
        while i < tlen:
            pi = tpos + i
            c = self.text.U8(pi)
            if c == 0:
                if not i == tlen - 1:
                    report('warning', f'EOS at {i}/{tlen-1}')
                    i += 1
                    #continue
                else:
                    break
            elif c < 0x80:
                i += 1
            elif c < 0xe0:
                c2 = self.text.U8(pi + 1)
                c = (((c & 0x1f) << 6) | (c2 & 0x3f))
                i += 2
            elif c < 0xf0:
                c2 = self.text.U8(pi + 1)
                c3 = self.text.U8(pi + 2)
                c = (((c & 0x1f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f))
                i += 3
            else:
                i += 1
            if mark_ctrl and c >= fmx:
                c |= 0xf000
            r.append(c)
        else:
            report('warning', f'something wrong when decode text 0x{tidx:x}')
        return r

    def enc_text(self, txt):
        raise NotImplementedError

class c_ff5a_parser_text_cn(c_ff5a_parser_text):

    def dec_text(self, tpos, tlen, tidx, mark_ctrl = True):
        r = []
        i = 0
        while i < tlen:
            pi = tpos + i
            c = self.text.U8(pi)
            if c == 0:
                if not i == tlen - 1:
                    report('warning', f'EOS at {i}/{tlen-1}')
                    i += 1
                    #continue
                else:
                    break
            elif c < 0xe0:
                i += 1
            elif c < 0xf0:
                c2 = self.text.U8(pi + 1)
                c = (((c - 0xdf) << 8 ) | c2) - 0x20
                i += 2
            else:
                #ctrl sym
                c2 = self.text.U8(pi + 1)
                if not mark_ctrl:
                    c &= 0xf
                c = ((c << 8) | c2)
                i += 2
            r.append(c)
        else:
            report('warning', f'something wrong when decode text 0x{tidx:x}')
        return r

    def enc_text(self, txt):
        r = []
        tlen = len(txt)
        for i, c in enumerate(txt):
            if c == 0:
                if not i == tlen - 1:
                    report('warning', f'EOS at {i}/{tlen-1}')
                    r.append(c)
                else:
                    break
            elif c < 0xe0:
                r.append(c)
            elif c < 0x11e0:
                c += 0x20
                r.append((c >> 8) + 0xdf)
                r.append(c & 0xff)
            elif 0xf000 <= c < 0x10000:
                #ctrl sym
                r.append(c >> 8)
                r.append(c &0xff)
            else:
                report('warning', f'invalid char {c:x}, ignored')
        else:
            r.append(0)
            #report('warning', f'EOS missing, auto filled')
        return r

class c_ff5a_parser:

    def __init__(self, path):
        self.rom_path = path
        self.txt_parser = {}

    def load_rom(self):
        with open(self.rom_path, 'rb') as fd:
            raw = fd.read()
        self.rom = c_ff5a_sect_rom(raw, 0)
        self.rom.parse()

    def new_txt_parser(self, name, txt_idx, fnt_idx, cls):
        self.txt_parser[name] = cls(
            self.rom.tabs['text'][txt_idx],
            self.rom.tabs['font'][fnt_idx])

    @property
    def one_txt_parser(self):
        for k in self.txt_parser:
            return self.txt_parser[k]
        return None

    def draw_txt_parser(self, lines, cols = 10, mkimg = True):
        opsr = self.one_txt_parser
        tdr = opsr.draw
        if not tdr:
            return None
        vblks = []
        for tidx in lines:
            hblks = [opsr.draw_comment(tidx)]
            for nm, tpsr in self.txt_parser.items():
                hblks.append(tpsr.draw_text(tidx, cols))
            if hblks:
                vblks.append(tdr.draw_horiz(*hblks, pad = 20))
        blk = tdr.draw_vert(*vblks)
        if mkimg:
            return tdr.make_img(blk)
        else:
            return blk

    def draw_txt_chars(self, name, chars, cols = 16, mkimg = True):
        tpsr = self.txt_parser[name]
        blk = tpsr.draw_chars(chars, cols)
        if mkimg:
            return tpsr.draw.make_img(blk)
        else:
            return blk

    def find_txt_chars(self, name, chars, first = True, strict = False, ret_txt = False):
        return self.txt_parser[name].find_chars(chars, first, strict, ret_txt)

    def repack_txt_with(self, name, rplc):
        tpsr = self.txt_parser[name]
        txtmk = tpsr.repack_text(rplc)
        rmk = self.rom.repack_with(tpsr.text, txtmk)
        return rmk

    def parse(self):
        self.new_txt_parser('jp', 0, 0, c_ff5a_parser_text_jp)
        self.new_txt_parser('cn', 1, 4, c_ff5a_parser_text_cn)
##        self.txt_ctrl_set = {}
##        self.txt_ctrl_set['jp'] = self.count_ctrl_sym('jp')
##        self.txt_ctrl_set['cn'] = self.count_ctrl_sym('cn')

    def count_ctrl_sym(self, name):
        tpsr = self.txt_parser[name]
        ctrl_set = set()
        for i in range(tpsr.text.cnt_index):
            t = tpsr.get_text(i)
            for c in t:
                if tpsr.is_ctrl(c):
                    ctrl_set.add(c)
        return sorted(ctrl_set)

    def count_uni_chars(self, name, skip_even = False):
        tpsr = self.txt_parser[name]
        mul = set()
        uni = {}
        uni_r = {}
        if skip_even:
            trng = range(1, tpsr.text.cnt_index, 2)
        else:
            trng = range(tpsr.text.cnt_index)
        for i in trng:
            t = tpsr.get_text(i)
            for c in t:
                if c in uni:
                    _i = uni[c]
                    _u = uni_r[_i]
                    _u.remove(c)
                    if not _u:
                        del uni_r[_i]
                    del uni[c]
                    mul.add(c)
                elif c in mul:
                    pass
                else:
                    uni[c] = i
                    if not i in uni_r:
                        uni_r[i] = []
                    uni_r[i].append(c)
        return uni_r

    def draw_txt_uni_chars(self, name, skip_even = False,
            cols1 = 5, cols2 = 10, mkimg = True):
        ucinfo = self.count_uni_chars(name, skip_even)
        tpsr = self.txt_parser[name]
        tdr = tpsr.draw
        vblks = []
        for ti, ucs in ucinfo.items():
            vblks.append(tdr.draw_horiz(
                tpsr.draw_comment(ti),
                tpsr.draw_chars(ucs, cols1),
                tpsr.draw_text(ti, cols2),
            ))
        blk = tdr.draw_vert(*vblks)
        if mkimg:
            return tdr.make_img(blk)
        else:
            return blk

    def cmp_ctrl_sym(self, ignore = []):
        tpsrs = []
        for nm, tpsr in self.txt_parser.items():
            tpsrs.append(tpsr)
        r1 = []
        r2 = []
        plen = len(tpsrs)
        for ti in range(1, tpsrs[0].text.cnt_index, 2):
            cseqs = []
            csets = []
            csa = set()
            skip_empty = False
            for i, tpsr in enumerate(tpsrs):
                t = tpsr.get_text(ti)
                if not t:
                    t = tpsr.get_text(ti - 1)
                cseq = []
                cset = {}
                for c in t:
                    if not tpsr.is_ctrl(c) or c in ignore:
                        continue
                    cseq.append(c)
                    if not c in cset:
                        cset[c] = 0
                    cset[c] += 1
                    csa.add(c)
                cseqs.append(cseq)
                csets.append(cset)
            for cs in csa:
                ccnt = min(cset[cs] if cs in cset else 0 for cset in csets)
                if ccnt > 0:
                    for cset in csets:
                        cset[cs] -= ccnt
                        if cset[cs] < 1:
                            del cset[cs]
            if any(csets):
                r1.append((ti, csets))
                continue
            idxs = [0] * plen
            cnt_eos = 0
            while True:
                cur_cs = None
                unmatch = False
                for i in range(plen):
                    cseq = cseqs[i]
                    idx = idxs[i]
                    if idx >= len(cseq):
                        cnt_eos += 1
                        continue
                    c = cseq[idx]
                    idxs[i] += 1
                    if i == 0:
                        cur_cs = c
                    elif c != cur_cs:
                        unmatch = True
                        break
                else:
                    if 0 < cnt_eos < plen:
                        unmatch = True
                if unmatch:
                    r2.append((ti, cseqs))
                    break
                elif cnt_eos == plen:
                    break
        return r1, r2

    TXT_FNTCN_JPRNG = (0x99, 0x13b)
    def is_jp(self, c):
        r0, r1 = self.TXT_FNTCN_JPRNG
        return r0 <= c < r1
    
    def guess_non_trans_text(self):
        tpsr = self.txt_parser['cn']
        r = []
        for i in range(1, tpsr.text.cnt_index, 2):
            t = tpsr.get_text(i)
            if len(t) < 2:
                continue
            for c in t:
                if self.is_jp(c):
                    r.append(i)
                    break
        return r

    def guess_ctrl_fault(self):
        r1, r2 = self.cmp_ctrl_sym({0xf4e0, 0xf509})
        return sorted((*(i for i, cs in r1), *(i for i, cs in r2)))

    def guess_omit(self):
        tpsr = self.txt_parser['cn']
        r = []
        for i in range(0, tpsr.text.cnt_index - 1, 2):
            t1 = tpsr.get_text(i)
            t2 = tpsr.get_text(i+1)
            if len(t1) > 0 and len(t2) == 0:
                r.append(i)
        return r

if __name__ == '__main__':
    from hexdump import hexdump
    from pprint import pprint
    ppr = lambda *a, **ka: pprint(*a, **ka, sort_dicts = False)
    #psr = c_ff5a_parser('ff5ajp.gba')
    psr = c_ff5a_parser('ff5acn.gba')
    psr.load_rom()
    psr.parse()
    txtjp = psr.rom.tabs['text'][0]
    txtcn = psr.rom.tabs['text'][1]
    fntjp = psr.rom.tabs['font'][0]
    fntcn = psr.rom.tabs['font'][4]
    def _show_long_txt(txt, th = 8):
        for i in range(txt.cnt_index):
            s, p, l = txt.get_raw(i, True)
            if l > th:
                print(f'{i}:+0x{p:x}(0x{l:x})')
                hexdump(s)
                yield
    slt = _show_long_txt(txtjp, 0x20)
    def _cmp_long_txt(th):
        tcnt = min(txtjp.cnt_index, txtcn.cnt_index)
        print(f'jp:{txtjp.cnt_index}, cn:{txtcn.cnt_index}')
        for i in range(tcnt):
            s1, p1, l1 = txtjp.get_raw(i, True)
            s2, p2, l2 = txtcn.get_raw(i, True)
            if l1 == l2 > th:
                print(f'jp {i}:+0x{p1:x}(0x{l1:x})')
                hexdump(s1)
                print(f'cn {i}:+0x{p2:x}(0x{l2:x})')
                hexdump(s2)
                yield
    clt = _cmp_long_txt(0x20)
    def _cmp_font(n):
        if fntjp.font_size != fntcn.font_size:
            return
        fcnt = min(fntjp.cnt_index, fntcn.cnt_index)
        print(f'jp:0x{fntjp.cnt_index:x}, cn:0x{fntcn.cnt_index:x}')
        rs = []
        rc = 0
        for i in range(fcnt):
            f1, p1, l1 = fntjp.get_raw(i, True)
            f2, p2, l2 = fntcn.get_raw(i, True)
            if l1 != l2:
                print(f'unmatch size {i:04x}:{l1}/{l2}')
                continue
            for j in range(l1):
                if f1[j] != f2[j]:
                    break
            else:
                if rc < th:
                    rs.append(i)
                    rc += 1
                else:
                    r = ' '.join(f'{c:04x}' for c in rs)
                    print(f'same: {r}')
                    rs = []
                    rc = 0
                    yield
        if rc:
            r = ' '.join(f'{c:04x}' for c in rs)
            print(f'same: {r}')
        yield
    cf = _cmp_font(8)
    #next(cf)
    #next(clt)
    def _count_font(txt):
        fntset = set()
        for i in range(txt.cnt_index):
            s, p, l = txt.get_raw(i, True)
            for i in range(0, l-1, 2):
                c = s[i] + (s[i+1] << 8)
                fntset.add(c)
        return fntset
##    dtjp = c_text_drawer(fntjp)
##    dtcn = c_text_drawer(fntcn)
##    im = dtjp.make_img(
##        dtjp.draw_vert(
##            dtjp.draw_block([10, 11, 12, 13, 14, 15, 16, 17, 18], 3),
##            dtjp.draw_horiz(
##                dtjp.draw_comment(30, 10, '123'),
##                dtjp.draw_block([10, 11, 12, 13, 14, 15, 16, 17, 18], 5),
##                dtjp.draw_block([10, 11, 12, 13, 14, 15, 16, 17, 18], 3),
##            ),
##        )
##    )
##    psr.new_txt_parser('jp', 0, 0, c_ff5a_parser_text_jp)
##    psr.new_txt_parser('cn', 1, 4, c_ff5a_parser_text_cn)
    tpjp = psr.txt_parser['jp']
    tpcn = psr.txt_parser['cn']
    def _count_ctrl_sym(tp):
        cs = set()
        for i in range(tp.text.cnt_index):
            t = tp.get_text(i)
            for c in t:
                if c > 0x4df:
                    cs.add(c)
        return sorted(cs)
    #csjp = _count_ctrl_sym(tpjp)
    #cscn = _count_ctrl_sym(tpcn)
    def _find_non_ctrl():
        r = {}
        for i in range(tpjp.text.cnt_index):
            tjp = tpjp.get_text(i)
            tcn = tpcn.get_text(i)
            cc = {}
            for c in tjp:
                if c in csjp:
                    if not c in cc:
                        cc[c] = 0
                    cc[c] += 1
            cc2 = {}
            for c in tcn:
                if c in csjp:
                    if c in cc:
                        if cc[c] > 1:
                            cc[c] -= 1
                        else:
                            del cc[c]
                    else:
                        if not c in cc2:
                            cc2[c] = 0
                        cc2[c] += 1
            if cc or cc2:
                r[i] = (cc, cc2)
        return r
##    ccmp = _find_non_ctrl()
##    im = psr.draw_txt_parser([*ccmp.keys()][:30])
##    im.save('out.png')
##    for i, (k, (cc1, cc2)) in enumerate(ccmp.items()):
##        if i >= 30:
##            break
##        print(k, {hex(i): v for i, v in cc1.items()}, {hex(i): v for i, v in cc2.items()})
    def _find_non_trans():
        r = []
        for i in range(1, tpjp.text.cnt_index, 2):
            tjp = tpjp.get_text(i)
            tcn = tpcn.get_text(i)
            if not len(tjp) == len(tcn) > 1:
                continue
            for j in range(len(tjp)):
                cjp = tjp[j]
                if not cjp in csjp:
                    continue
                ccn = tcn[j]
                #if not (ccn & 0xf000):
                #    break
                #ccn &= 0xfff
                if not cjp == ccn:
                    break
            else:
                r.append(i)
        return r
    #tnt = _find_non_trans()
    #tnt = psr.guess_non_trans_text()
    #im = psr.draw_txt_parser(tnt[:])
    #im.save('out.png')
    #gcsf = psr.guess_ctrl_fault()
    #im = psr.draw_txt_parser([v for i in gcsf for v in (i-3, i-2, i-1, i, i+1, i+2)])
    #im = psr.draw_txt_parser([v for i in gcsf for v in (i-1, i)])
    #im.save('occs2.png')
    
