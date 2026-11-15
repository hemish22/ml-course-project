#!/bin/sh
# Rebuild report/ML_Project_Report.docx from the template and saved results, then use
# LibreOffice (headless) to refresh the table of contents / figure and table lists and
# export report/ML_Project_Report.pdf. Usage: scripts/render_report_pdf.sh "<template.docx>"
set -e
cd "$(dirname "$0")/.."
SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
PROFILE=/tmp/lo_profile_report
.venv/bin/python scripts/build_report.py --template "$1"
if [ ! -d "$PROFILE/user/basic/Standard" ]; then
  "$SOFFICE" -env:UserInstallation=file://$PROFILE --headless --terminate_after_init
fi
cat > "$PROFILE/user/basic/Standard/Module1.xba" <<'XBA'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">
Sub UpdateAndExport(src As String, pdf As String)
  Dim args(0) As New com.sun.star.beans.PropertyValue
  args(0).Name = &quot;Hidden&quot;
  args(0).Value = True
  doc = StarDesktop.loadComponentFromURL(ConvertToURL(src), &quot;_blank&quot;, 0, args())
  doc.getTextFields().refresh()
  idx = doc.getDocumentIndexes()
  For k = 1 To 2
    For i = 0 To idx.getCount() - 1
      idx.getByIndex(i).update()
    Next i
  Next k
  Dim out(0) As New com.sun.star.beans.PropertyValue
  out(0).Name = &quot;FilterName&quot;
  out(0).Value = &quot;writer_pdf_Export&quot;
  doc.storeToURL(ConvertToURL(pdf), out())
  doc.close(True)
End Sub
</script:module>
XBA
P="$PWD/report"
"$SOFFICE" -env:UserInstallation=file://$PROFILE --headless --invisible \
  "macro:///Standard.Module1.UpdateAndExport(\"$P/ML_Project_Report.docx\",\"$P/ML_Project_Report.pdf\")" 2>/dev/null
echo "Saved $P/ML_Project_Report.pdf"
