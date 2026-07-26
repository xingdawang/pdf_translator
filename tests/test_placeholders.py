from pdf_translator.placeholders import PlaceholderService


def test_placeholder_protect_restore_and_tolerant_brackets():
    service = PlaceholderService()
    source = (
        "Email qa@example.org, visit https://example.com/a and use 25 mg "
        "with formula $E=mc^2$ [12]."
    )
    result = service.protect(source, "task", "P0001-S000001")
    assert len(result.placeholders) >= 5
    assert any(item.kind == "formula" for item in result.placeholders)
    assert all(item.original not in result.protected_text for item in result.placeholders)

    translated = "中文：" + result.protected_text.replace("⟦", "【 ").replace("⟧", " 】")
    restored = service.restore(translated, result.placeholders)
    assert restored.ok
    for placeholder in result.placeholders:
        assert placeholder.original in restored.restored_text


def test_placeholder_duplicate_is_blocking_signal():
    service = PlaceholderService()
    result = service.protect(
        "Read https://example.com now.", "task", "P0001-S000001"
    )
    token = result.placeholders[0].token
    restored = service.restore(result.protected_text + " " + token, result.placeholders)
    assert not restored.ok
    assert restored.duplicated == [token]


def test_fraction_fragment_and_decade_are_not_protected_as_path_or_unit():
    service = PlaceholderService()

    result = service.protect(
        "A knife scratches under 5*/2). This became common in the 1990s.",
        "task",
        "segment",
    )

    assert not any(
        placeholder.original in {"/2).", "1990s"}
        for placeholder in result.placeholders
    )


def test_output_policy_allows_changed_number_unit_but_not_missing_url():
    service = PlaceholderService()
    result = service.protect(
        "Visit https://example.com and use 25 mg.",
        "task",
        "segment",
    )
    restored = service.restore("请访问网站并使用 25 毫克。", result.placeholders)

    assert not service.output_restore_ok(
        restored,
        result.placeholders,
        ignore_number_warnings=True,
    )
    url_placeholder = next(
        item for item in result.placeholders if item.kind == "url"
    )
    number_placeholder = next(
        item for item in result.placeholders if item.kind == "number_unit"
    )
    assert number_placeholder.token in service.allowed_missing_tokens(
        restored,
        result.placeholders,
        ignore_number_warnings=True,
    )
    assert url_placeholder.token in service.blocking_missing_tokens(
        restored,
        result.placeholders,
        ignore_number_warnings=True,
    )


def test_manual_repair_can_insert_original_value_directly():
    service = PlaceholderService()
    result = service.protect(
        "Read https://example.com now.", "task", "P0001-S000001"
    )
    restored = service.restore(
        "请立即阅读 https://example.com。", result.placeholders
    )
    assert restored.ok
    assert "https://example.com" in restored.restored_text


def test_tokens_are_stable_and_unique_within_task():
    service = PlaceholderService()
    first = service.protect(
        "Use 10 mg and 20 mg.", "task", "P0001-S000001"
    )
    second = service.protect(
        "Use 10 mg and 20 mg.", "task", "P0001-S000002"
    )
    assert [item.token for item in first.placeholders] == [
        item.token
        for item in service.protect(
            "Use 10 mg and 20 mg.", "task", "P0001-S000001"
        ).placeholders
    ]
    assert set(item.token for item in first.placeholders).isdisjoint(
        item.token for item in second.placeholders
    )
