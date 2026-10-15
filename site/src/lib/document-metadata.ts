import { useEffect } from "react";

export interface DocumentMetadata {
  description: string;
  title: string;
}

export function useDocumentMetadata({ description, title }: DocumentMetadata): void {
  useEffect(() => {
    document.title = title;

    let descriptionMeta = document.querySelector<HTMLMetaElement>('meta[name="description"]');
    if (!descriptionMeta) {
      descriptionMeta = document.createElement("meta");
      descriptionMeta.name = "description";
      document.head.append(descriptionMeta);
    }
    descriptionMeta.content = description;
  }, [description, title]);
}
