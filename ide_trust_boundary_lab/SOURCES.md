# Sources, checked November 7, 2026

1. Cursor, October 29 notice, with updated IOCs (https://cursor.com/en-US/blog/claude-code-extension-security-notice).
2. Cursor, Extensions documentation, cooldown/signatures (https://cursor.com/help/customization/extensions).
3. StepSecurity, October 30 / November 1 static analysis (https://www.stepsecurity.io/blog/claude-code-extension-trojanized).
4. IBM, Langflow OSS multiple vulnerabilities, Oct 28/29 (https://www.ibm.com/support/pages/security-bulletin-langflow-oss-affected-multiple-vulnerabilities-2).
5. Langflow 1.13 announcement and upgrade notes (https://www.langflow.org/blog/langflow-1-13/), (https://docs.langflow.org/troubleshoot).
6. Langflow GitHub releases, 1.12.9/1.12.10 and 1.13.1 release notes (https://github.com/langflow-ai/langflow/releases).
7. Johann Rehberger, October 13 research disclosure (https://embracethered.com/blog/posts/2026/visual-studio-copilot-remote-code-execution/).
8. Microsoft MSRC, CVE-2026-103572 (https://msrc.microsoft.com/update-guide/vulnerability/CVE-2026-103572).

**Important corrections:** Up to 5,000 is an exposure estimate, not verified compromises. The vendor has not disclosed Cursor's original compromise vector. A 12-hour cooldown is a mitigation, not a universal guarantee. Cursor's current documentation provides optional enterprise signature verification, while its older security page states verification was not enabled by default. Langflow 1.12.9/1.12.10 were released after IBM's advisory; the bulletin does not certify those backports. Version sorting alone does not demonstrate remediation.
