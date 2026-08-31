from django import forms


class DocumentUploadForm(forms.Form):
    file = forms.FileField(
        label="Fichier TXT",
        help_text="Format accepté : texte UTF-8 (.txt).",
    )

    def clean_file(self):
        uploaded_file = self.cleaned_data["file"]

        if not uploaded_file.name.lower().endswith(".txt"):
            raise forms.ValidationError(
                "Seuls les fichiers TXT sont actuellement pris en charge."
            )

        return uploaded_file