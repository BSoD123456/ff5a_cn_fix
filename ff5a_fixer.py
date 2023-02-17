#! python3
# coding: utf-8

import os, os.path
import json

from ff5a_parser import c_ff5a_parser

def report(*args):
    r = ' '.join(args)
    print(r)
    return r

class c_ff5a_fixer:

    def __init__(self, paths):
        self.paths = paths

    def load(self):
        self.load_rom()
        self.load_charset()

    def load_rom(self):
        path = self.paths['rom']
        self.psr = c_ff5a_parser(path)
        self.psr.load_rom()
        self.psr.parse()

    def load_charset(self):
        path = self.paths['charset']
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                cs, csr = json.load(fd)
            self.chst = [{int(k): v for k, v in c.items()} for c in cs[:2]]
            self.chst_r = csr[:2]
        except:
            self.update_charset()

    def update_charset(self):
        from ff5a_ocr import c_ff5a_ocr_parser
        ocr = c_ff5a_ocr_parser(self.psr.txt_parser['cn'])
        ocr.parse()
        ocr.feed_all()
        cs, csr = ocr.export_charset()
        self.chst = cs[:2]
        self.chst_r = csr[:2]
        path = self.paths['charset']
        try:
            with open(path, 'w', encoding='utf-8') as fd:
                json.dump((cs, csr), fd, ensure_ascii=False)
        except:
            pass

    def chr(self, i):
        tpsr = self.psr.txt_parser['cn']
        cc = tpsr.dec_ctrl(i)
        if cc:
            return f'[C{cc}]'
        for ci, cs in enumerate(self.chst):
            if i in cs:
                c = cs[i]
                if ci > 0:
                    report('warning', f'non-determine char {c}({i:x})')
                return c
        return f'[U{i:04x}]'

    def ord(self, c):
        for ci, csr in enumerate(self.chst_r):
            if c in csr:
                i = csr[c]
                if ci > 0:
                    report('warning', f'non-determine char {c}({i:x})')
                return i

    def toloc(self, s):
        return [self.ord(c) for c in s]

    def tostr(self, s):
        return ''.join(self.chr(i) for i in s)

    def get_text(self, tidx, name = 'cn'):
        src = self.psr.txt_parser[name].get_text(tidx)
        return self.tostr(src)

    def repack_with(self, rplc):
        mk = self.psr.repack_txt_with('cn',
            {i:self.toloc(s) for i, s in rplc.items()})
        with open(self.paths['rom_out'], 'wb') as fd:
            fd.write(mk.BYTES(0, None))

if __name__ == '__main__':

    from pprint import pprint
    ppr = lambda *a, **ka: pprint(*a, **ka, sort_dicts = False)

    FF5A_PATHS = {
        'rom': 'ff5acn.gba',
        'rom_out': 'ff5acn_out.gba',
        'charset': 'charset.json',
    }
    fx = c_ff5a_fixer(FF5A_PATHS)
    fx.load()
