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

    def __init__(self, paths, dump_info):
        self.paths = paths
        self.info_dump = dump_info

    def load(self):
        self.load_rom()
        self.load_charset()
        self.load_patch()

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

    def load_patch(self):
        path = self.paths['patch']
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                patch = json.load(fd)
            self.patch = {d: {int(k): v for k, v in b.items()} for d, b in patch.items()}
        except:
            self.reset_patch()

    def reset_patch(self):
        path = self.paths['patch']
        patch = self.dump_with(self.info_dump)
        try:
            with open(path, 'w', encoding='utf-8') as fd:
                json.dump(patch, fd,
                    ensure_ascii=False,indent=4, sort_keys=False)
        except:
            pass
        self.patch = patch

    def chr(self, i):
        for ci, cs in enumerate(self.chst):
            if i in cs:
                c = cs[i]
                if ci > 0:
                    report('warning', f'non-determine char {c}({i:x})')
                return c

    def ord(self, c):
        for ci, csr in enumerate(self.chst_r):
            if c in csr:
                i = csr[c]
                if ci > 0:
                    report('warning', f'non-determine char {c}({i:x})')
                return i

    def parse(self, s):
        r = []
        ci = [-1]
        slen = len(s)
        ctx = {}
        def gc(la):
            i = ci[-1] + la
            if 0 <= i < slen:
                return s[i]
            else:
                return None
        def stp(v):
            i = ci[-1]
            ri = min(slen, i + v)
            ci[-1] = ri
            return ri
        def psh():
            ci.append(ci[-1])
        def dsc():
            return ci.pop()
        def mrg():
            ci[-1] = dsc()
        while stp(1) < slen:
            c = gc(0)
            if c == '[':
                psh()
                cmd = []
                cmd_valid = False
                while stp(1) < slen:
                    la = gc(0).lower()
                    if la == '[':
                        break
                    elif la == ']':
                        cmd_valid = self._parse_cmd(cmd, ctx)
                        break
                    cmd.append(la)
                if cmd_valid:
                    mrg()
                    rc = ctx.pop('ret', None)
                    if not rc is None:
                        if isinstance(rc, list):
                            r.extend(rc)
                        else:
                            r.append(rc)
                    if ctx.pop('break', False):
                        break
                    else:
                        continue
                else:
                    dsc()
            rc = self.ord(c)
            if rc is None:
                breakpoint()
                report('warning', f'unknown char {c} at {ci[0]}, ignored')
            else:
                r.append(rc)
        return r, ctx

    def _parse_cmd(self, cmd, ctx):
        if not cmd:
            return False
        [c, *v] = cmd
        v = ''.join(v)
        try:
            if c == 'c':
                tpsr = self.psr.txt_parser['cn']
                ctx['ret'] = tpsr.enc_ctrl(int(v))
            elif c == 'u':
                ctx['ret'] = int(v, base=16)
            elif c == 'f' and v[0] == ':':
                if not 'flags' in ctx:
                    ctx['flags'] = set()
                ctx['flags'].add(v[1:])
                ctx['break'] = True
            elif c == '#':
                # comment
                pass
            else:
                raise
        except:
            report('warning', f'cmd[{c}{v}] failed')
            return False
        return True

    def toloc(self, s):
        return self.parse(s)[0]

    def tostr(self, s):
        tpsr = self.psr.txt_parser['cn']
        r = []
        for i in s:
            cc = tpsr.dec_ctrl(i)
            if cc:
                c = f'[C{cc}]'
            else:
                c = self.chr(i)
                if c is None:
                    c = f'[U{i:04x}]'
            r.append(c)
        return ''.join(r)

    def get_text(self, tidx, name = 'cn'):
        src = self.psr.txt_parser[name].get_text(tidx)
        return self.tostr(src)

    def find_text(self, src, name = 'cn'):
        return [ti for ti, ci in self.psr.find_txt_chars(name, fx.toloc(src), False)]

    def dump_texts(self, blk, tidxs, surplus = 0, skipeven = True):
        for tidx in tidxs:
            if surplus > 0:
                srng = (-(surplus-1)*2-(tidx%2), surplus*2-(tidx%2))
            else:
                srng = (0, 1)
            for i in range(*srng):
                ti = tidx + i
                if skipeven and not ti % 2:
                    continue
                txt = self.get_text(ti)
                ptxt = (self.get_text(ti-1) if ti > 0 else '')
                if not (txt or ptxt):
                    continue
                ptxt = '[#prev]' + ptxt
                if not txt and ti % 2:
                    txt = ptxt
                    is_prev = True
                else:
                    is_prev = False
                if i == 0:
                    if is_prev:
                        blk[ti] = {txt: '[#dump]' + txt}
                    else:
                        blk[ti] = {txt: ''}
                else:
                    if ti in blk:
                        continue
                    if skipeven:
                        blk[ti] = {txt: '[F:refer]' + ptxt}
                    else:
                        blk[ti] = {txt: '[F:refer]'}

    def dump_with(self, info):
        blks = {}
        wlk = set()
        for desc, cb in info.items():
            blk = {}
            for ti_info in cb(self):
                tidxs = []
                for ti in ti_info[0]:
                    if ti in wlk:
                        continue
                    wlk.add(ti)
                    tidxs.append(ti)
                self.dump_texts(blk, tidxs, *ti_info[1:])
            blks[desc] = blk
        return blks

    def repack_replace(self, blks):
        rplc = {}
        for desc, blk in blks.items():
            for ti, tr in blk.items():
                assert(len(tr)) == 1
                for stxt, ctxt in tr.items():
                    pass
                dtxt, ctx = self.parse(ctxt)
                if 'flags' in ctx:
                    flgs = ctx['flags']
                    if 'clear' in flgs:
                        rplc[ti] = ''
                        report('info', f'clear text({ti})\n{stxt}')
                        continue
                    elif 'refer' in flgs:
                        continue
                if dtxt:
                    report('info', f'replace text({ti})\n{stxt} -> {self.tostr(dtxt)}')
                    rplc[ti] = dtxt
        report('info', f'replace {len(rplc)} texts')
        return rplc

    def repack(self):
        rplc = self.repack_replace(self.patch)
        mk = self.psr.repack_txt_with('cn', rplc)
        with open(self.paths['rom_out'], 'wb') as fd:
            fd.write(mk.BYTES(0, None))

if __name__ == '__main__':

    from pprint import pprint
    ppr = lambda *a, **ka: pprint(*a, **ka, sort_dicts = False)

    FF5A_PATHS = {
        'rom': 'ff5acn.gba',
        'rom_out': 'ff5acn_out.gba',
        'charset': 'charset.json',
        'patch': 'patch.json',
    }
    FF5A_DUMP_INFO = {
        'wrong': lambda fx: [
            (fx.find_text('消息速度'), 1, False),
        ],
        'name table': lambda fx: [
            (range(450, 472),),
        ],
        'omit': lambda fx: [
            ([i+1 for i in fx.psr.guess_omit() if not i+1 in {
                9109, 9207, 9239, 9243, 9245, 9265,
                9027, # not sure
            }], 3),
        ],
    }
    fx = c_ff5a_fixer(FF5A_PATHS, FF5A_DUMP_INFO)
    fx.load()
