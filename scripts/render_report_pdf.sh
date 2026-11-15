#!/bin/sh
# Rebuild report/ML_Project_Report.docx from the cover (report/cover.docx) and saved results,
# use LibreOffice (headless) to paginate it, write the page numbers into the docx's contents
# and lists, and export report/ML_Project_Report.pdf. Usage: scripts/render_report_pdf.sh [cover.docx]
set -e
cd "$(dirname "$0")/.."
SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
PROFILE=/tmp/lo_profile_report
.venv/bin/python scripts/build_report.py ${1:+--template "$1"}
if [ ! -d "$PROFILE/user/basic/Standard" ]; then
  "$SOFFICE" -env:UserInstallation=file://$PROFILE --headless --terminate_after_init
fi
cat > "$PROFILE/user/basic/Standard/Module1.xba" <<'XBA'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">
Sub UpdateAndExport(src As String, pdf As String, refresh As Boolean)
  Dim args(0) As New com.sun.star.beans.PropertyValue
  args(0).Name = &quot;Hidden&quot;
  args(0).Value = True
  doc = StarDesktop.loadComponentFromURL(ConvertToURL(src), &quot;_blank&quot;, 0, args())
  doc.getTextFields().refresh()
  If refresh Then
    idx = doc.getDocumentIndexes()
    For k = 1 To 2
      For i = 0 To idx.getCount() - 1
        idx.getByIndex(i).update()
      Next i
    Next k
  End If
  Dim out(0) As New com.sun.star.beans.PropertyValue
  out(0).Name = &quot;FilterName&quot;
  out(0).Value = &quot;writer_pdf_Export&quot;
  doc.storeToURL(ConvertToURL(pdf), out())
  doc.close(True)
End Sub
</script:module>
XBA
P="$PWD/report"
export_pdf() {  # $1 = true: let LibreOffice rebuild the indexes (first pass only)
  "$SOFFICE" -env:UserInstallation=file://$PROFILE --headless --invisible \
    "macro:///Standard.Module1.UpdateAndExport(\"$P/ML_Project_Report.docx\",\"$P/ML_Project_Report.pdf\",$1)" 2>/dev/null
}
fill() { .venv/bin/python scripts/fill_report_indexes.py "$P/ML_Project_Report.docx" "$P/ML_Project_Report.pdf"; }
# Pass 1 paginates the body; later passes keep the stored (compact) entries and settle the front-matter numbers.
export_pdf true; fill; export_pdf false; fill; export_pdf false
echo "Saved $P/ML_Project_Report.pdf"
