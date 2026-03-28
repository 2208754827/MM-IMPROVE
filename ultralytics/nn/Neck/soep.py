# Ultralytics YOLOMM - SOEP (Small Object Enhance Pyramid) Module
# SOEP: 灏忕洰鏍囧寮洪噾瀛楀妯″潡

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from ultralytics.nn.modules import Conv

__all__ = ['SPDConv', 'FGM', 'OmniKernel', 'CSPOmniKernel']


class SPDConv(nn.Module):
    """Space-to-Depth Convolution - 绌洪棿鍒版繁搴﹀嵎绉?

    灏嗙┖闂寸淮搴︾殑淇℃伅鍘嬬缉鍒伴€氶亾缁村害锛岄伩鍏嶄紶缁熶笅閲囨牱瀵艰嚧鐨勪俊鎭涪澶便€?    鐗瑰埆閫傜敤浜庝繚鐣欏皬鐩爣鐨勭粏鑺備俊鎭€?
    璁捐鍘熺悊:
        灏?x2鐨勭┖闂村尯鍩熼噸缁勪负4涓€氶亾锛屼娇寰?
        - 绌洪棿鍒嗚鲸鐜囬檷浣?鍊?        - 閫氶亾鏁板鍔?鍊?        - 淇℃伅瀹屽叏淇濈暀锛屾棤鎹熷け

    Args:
        inc: 杈撳叆閫氶亾鏁?        ouc: 杈撳嚭閫氶亾鏁?        dimension: 缁村害鍙傛暟锛堥粯璁や负1锛?
    Examples:
        >>> spdconv = SPDConv(256, 128)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = spdconv(x)  # shape: (1, 128, 32, 32)
    """

    def __init__(self, inc, ouc, dimension=1):
        """Initialize SPDConv with input/output channels."""
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        """Apply space-to-depth transformation followed by convolution."""
        # 灏?x2鍖哄煙鐨?涓綅缃垎鍒彁鍙栧苟鎷兼帴鍒伴€氶亾缁村害
        x = torch.cat([
            x[..., ::2, ::2],   # 宸︿笂
            x[..., 1::2, ::2],  # 鍙充笂
            x[..., ::2, 1::2],  # 宸︿笅
            x[..., 1::2, 1::2]  # 鍙充笅
        ], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    """Frequency Gating Module - 棰戝煙闂ㄦ帶妯″潡.

    缁撳悎绌洪棿鍩熷拰棰戝煙淇℃伅锛岄€氳繃FFT鍙樻崲鍜岄棬鎺ф満鍒跺寮虹壒寰佽〃杈俱€?
    鎶€鏈壒鐐?
        - 棰戝煙闂ㄦ帶: 浣跨敤FFT杩涜棰戝煙鐗瑰緛澶勭悊
        - 鍙涔犲弬鏁? alpha鍜宐eta鎺у埗棰戝煙鍜岀┖闂村煙鐨勮瀺鍚堟潈閲?        - 娈嬪樊杩炴帴: 淇濇寔鍘熷淇℃伅鐨勫悓鏃跺寮洪鍩熺壒寰?
    Args:
        dim: 鐗瑰緛閫氶亾鏁?
    Reference:
        AAAI2024 - OmniKernel璁烘枃鐨勬牳蹇冪粍浠?        https://ojs.aaai.org/index.php/AAAI/article/view/27907
    """

    def __init__(self, dim):
        """Initialize FGM with learnable alpha and beta parameters."""
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        """Apply frequency gating mechanism."""
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 棰戝煙鍙樻崲
        with autocast(enabled=False):
            x1_f = x1.float()
            x2_f = x2.float()
            x2_fft = torch.fft.fft2(x2_f, norm='backward')
            out = x1_f * x2_fft
            out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward').abs()
        out = out.to(x.dtype)

        # 鍔犳潈铻嶅悎
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """OmniKernel - 鍏ㄦ柟浣嶅灏哄害鏍告ā鍧?

    闆嗘垚澶氬昂搴︽劅鍙楅噹銆侀鍩熷寮哄拰绌洪棿閫氶亾娉ㄦ剰鍔涚殑鍒涙柊妯″潡銆?
    鏍稿績鍒涙柊:
        1. 澶氬昂搴︽牳铻嶅悎: 1x31, 31x1, 31x31, 1x1 鍥涚涓嶅悓灏哄害鐨勫嵎绉牳
        2. 棰戝煙閫氶亾娉ㄦ剰鍔?FCA): FFT鍙樻崲 + 閫氶亾鍔犳潈
        3. 绌洪棿閫氶亾娉ㄦ剰鍔?SCA): 鑷€傚簲姹犲寲 + 閫氶亾璋冨埗
        4. 棰戝煙闂ㄦ帶妯″潡(FGM): 杩涗竴姝ュ寮洪鍩熺壒寰?
    鎶€鏈壒鐐?
        - 姘村钩鏍?1x31): 鎹曡幏姘村钩鏂瑰悜鐨勭壒寰?        - 鍨傜洿鏍?31x1): 鎹曡幏鍨傜洿鏂瑰悜鐨勭壒寰?        - 鍏ㄥ眬鏍?31x31): 鎹曡幏澶ц寖鍥翠笂涓嬫枃
        - 灞€閮ㄦ牳(1x1): 淇濈暀灞€閮ㄧ粏鑺?        - 娣卞害鍙垎绂? 鎵€鏈夊ぇ鏍搁兘浣跨敤groups=dim闄嶄綆璁＄畻閲?
    Args:
        dim: 杈撳叆杈撳嚭閫氶亾鏁?
    Reference:
        AAAI2024 - OmniKernel: Building Omni Kernel for Convolutional Neural Networks
        https://ojs.aaai.org/index.php/AAAI/article/view/27907
    """

    def __init__(self, dim):
        """Initialize OmniKernel with multi-scale kernels and attention mechanisms."""
        super().__init__()

        ker = 31
        pad = ker // 2

        # 杈撳叆杈撳嚭鍗风Н
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        # 澶氬昂搴︽繁搴﹀彲鍒嗙鍗风Н
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)  # 姘村钩
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)  # 鍨傜洿
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)           # 鍏ㄥ眬
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)               # 灞€閮?
        self.act = nn.ReLU()

        # 绌洪棿閫氶亾娉ㄦ剰鍔?(SCA)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 棰戝煙閫氶亾娉ㄦ剰鍔?(FCA)
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        """Apply multi-scale kernels with frequency and spatial attention."""
        out = self.in_conv(x)

        # === 棰戝煙閫氶亾娉ㄦ剰鍔?(FCA) ===
        x_att = self.fac_conv(self.fac_pool(out))
        with autocast(enabled=False):
            out_f = out.float()
            x_att_f = x_att.float()
            x_fft = torch.fft.fft2(out_f, norm='backward')
            x_fft = x_att_f * x_fft
            x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward').abs()
        x_fca = x_fca.to(out.dtype)

        # === 绌洪棿閫氶亾娉ㄦ剰鍔?(SCA) ===
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)

        # === 澶氬昂搴︽牳铻嶅悎 ===
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class CSPOmniKernel(nn.Module):
    """CSP-OmniKernel - 缁撳悎CSP鎬濇兂鐨凮mniKernel妯″潡.

    閲囩敤Cross Stage Partial (CSP) 缁撴瀯锛屽皢鐗瑰緛鍒嗕负涓や釜鍒嗘敮澶勭悊锛?    - OmniKernel鍒嗘敮 (25%): 缁忚繃OmniKernel澶勭悊锛岃幏寰楀灏哄害鍜岄鍩熷寮虹壒寰?    - Identity鍒嗘敮 (75%): 鐩存帴閫氳繃锛岄檷浣庤绠楅噺

    璁捐鍘熷垯:
        - 骞宠　鎬ц兘鍜屾晥鐜?        - 榛樿e=0.25锛屽嵆25%閫氶亾缁忚繃澶嶆潅澶勭悊
        - 淇濇寔姊害娴佸姩鎬?
    Args:
        dim: 杈撳叆杈撳嚭閫氶亾鏁?        e: OmniKernel鍒嗘敮鐨勯€氶亾姣斾緥锛堥粯璁?.25锛?
    Examples:
        >>> csp_ok = CSPOmniKernel(256, e=0.25)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = csp_ok(x)  # shape: (1, 256, 32, 32)
    """

    def __init__(self, dim, e=0.25):
        """Initialize CSP-OmniKernel with channel split ratio."""
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        """Apply CSP structure with OmniKernel."""
        # 鍒嗘敮鍒嗗壊
        ok_branch, identity = torch.split(
            self.cv1(x),
            [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))],
            dim=1
        )
        # 铻嶅悎杈撳嚭
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
