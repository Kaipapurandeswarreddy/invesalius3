import wx

import invesalius.project as prj
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class ExportTextureFormatDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(
            parent,
            title=_("Export 3D Surface from Volume Rendering"),
            style=wx.DEFAULT_DIALOG_STYLE,
        )

        self.surface_index = None
        self.format = "OBJ"
        self._init_ui()
        self._populate_surfaces()

    def _init_ui(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Surface selection
        hbox_surface = wx.BoxSizer(wx.HORIZONTAL)
        st_surface = wx.StaticText(panel, label=_("Surface:"))
        self.cb_surface = wx.ComboBox(panel, style=wx.CB_READONLY)
        hbox_surface.Add(st_surface, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox_surface.Add(self.cb_surface, proportion=1)
        vbox.Add(hbox_surface, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Format selection
        self.rb_format = wx.RadioBox(
            panel,
            label=_("Format:"),
            choices=[_("Wavefront OBJ"), _("VRML")],
            majorDimension=1,
            style=wx.RA_SPECIFY_COLS,
        )
        self.rb_format.SetSelection(0)
        vbox.Add(self.rb_format, flag=wx.EXPAND | wx.ALL, border=10)

        # Buttons
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)

        panel.SetSizer(vbox)
        vbox.Fit(panel)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(panel, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        self.SetSizer(main_sizer)
        main_sizer.Fit(self)
        self.Layout()

    def _populate_surfaces(self):
        project = prj.Project()
        for index in project.surface_dict:
            surface = project.surface_dict[index]
            self.cb_surface.Append(f"Surface {index} - {surface.name}", index)

        if self.cb_surface.GetCount() > 0:
            self.cb_surface.SetSelection(0)

    def GetValues(self):
        if self.cb_surface.GetSelection() == wx.NOT_FOUND:
            return None, None

        index = self.cb_surface.GetClientData(self.cb_surface.GetSelection())
        fmt = "OBJ" if self.rb_format.GetSelection() == 0 else "VRML"
        return index, fmt


class ExportTextureProgressDialog(wx.Dialog):
    def __init__(self, parent, title=_("Generating Surface Texture")):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE)
        self._init_ui()
        self._bind_events()

    def _init_ui(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.st_status = wx.StaticText(panel, label=_("Initializing..."))
        vbox.Add(self.st_status, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=15)

        self.gauge = wx.Gauge(panel, range=100, size=(300, 20))
        vbox.Add(self.gauge, flag=wx.EXPAND | wx.ALL, border=15)

        panel.SetSizer(vbox)
        vbox.Fit(self)
        self.Layout()

    def _bind_events(self):
        Publisher.subscribe(self.OnUpdateProgress, "Update texture export progress")

    def OnUpdateProgress(self, progress, status=""):
        wx.CallAfter(self._update_progress_threadsafe, progress, status)

    def _update_progress_threadsafe(self, progress, status=""):
        if status:
            self.st_status.SetLabel(status)
        self.gauge.SetValue(progress)

        if progress >= 100:
            self.EndModal(wx.ID_OK)
